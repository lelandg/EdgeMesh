"""Regenerate DA3 from a saved project without modifying project or original assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reuse-report', action='store_true', help='Recompute comparison from the previously generated mesh without inference.')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    for key, value in {'QT_QPA_PLATFORM':'offscreen', 'MPLBACKEND':'Agg', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'EDGEMESH_DATA_DIR':str(args.runtime_root)}.items():
        os.environ[key] = value
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import trimesh
    from data_contracts import normalized_inverse_depth, proportional_shape
    from depth_to_3d import DepthTo3D
    from model_store import ModelStore

    project = json.loads(args.project.read_text(encoding='utf-8'))
    project_folder = args.project.parent
    source = project_folder / project['assets']['source']['path']
    depth_pro_path = project_folder / project['assets']['mesh']['path']
    old_meshes = [p for p in (project_folder / 'assets').glob('*.ply') if p != depth_pro_path]
    if len(old_meshes) != 1:
        raise ValueError('Expected exactly one original DA3 mesh besides accepted Depth Pro mesh.')
    original_da3_path = old_meshes[0]
    record = next(r for r in project['session']['history'] if r['settings']['model'] == 'Depth Anything 3 Small')
    settings = record['settings']
    if project['session'].get('mask') is not None or settings['use_processed_image_enabled']:
        raise ValueError('This validation supports the reported unmasked original-image workflow only.')
    originals = [args.project, source, depth_pro_path, original_da3_path]
    hashes_before = {str(p): digest(p) for p in originals}
    args.output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    target = proportional_shape(image.shape, settings['resolution'])
    if target != (527, 700):
        raise ValueError(f'Unexpected reported target dimensions: {target}')
    if args.reuse_report:
        previous = json.loads(args.report.read_text(encoding='utf-8'))
        corrected_path = previous['metrics']['corrected_da3']['path']
        seconds = previous['elapsed_seconds']
        model_info = previous['model_info']
    else:
        pipeline = DepthTo3D('depth_anything_3_small', verbose=False,
            model_store=ModelStore(root=args.runtime_root, cache_dir=args.runtime_root / 'cache' / 'huggingface'),
            allow_download=False, device='cpu')
        start = time.perf_counter()
        corrected_path, background = pipeline.process_image(source, image_data=image,
            output_dir=args.output, target_size=target,
            smoothing_method=settings['smoothing_method'], flat_back=settings['flat_back_enabled'],
            grayscale_enabled=settings['grayscale_enabled'], edge_detection_enabled=settings['edge_detection_enabled'],
            invert_colors_enabled=settings['invert_colors_enabled'], depth_amount=settings['depth_amount'],
            depth_drop_percentage=settings['depth_drop_percentage'], project_on_original=settings['project_on_original'],
            background_removal=settings['drop_background_enabled'], background_tolerance=settings['background_tolerance'],
            background_color=settings['current_selected_color'] if settings['use_selected_color'] else None,
            progress=lambda stage: print(stage, flush=True))
        seconds = time.perf_counter() - start
        np.save(args.output / 'corrected-smoothed-depth.npy', pipeline.depth_map, allow_pickle=False)
        model_info = pipeline.model_info
    paths = {'original_da3': original_da3_path, 'corrected_da3': Path(corrected_path), 'original_depth_pro': depth_pro_path}
    meshes = {name: trimesh.load_mesh(str(path), process=False) for name, path in paths.items()}
    baseline = meshes['original_da3']
    metrics = {}
    for name, mesh in meshes.items():
        xy = np.asarray(mesh.vertices)[:, :2]
        same_xy = np.array_equal(xy, np.asarray(baseline.vertices)[:, :2])
        same_colors = same_xy and np.array_equal(mesh.visual.vertex_colors, baseline.visual.vertex_colors)
        unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        top = np.full(len(unique_xy), -np.inf)
        np.maximum.at(top, inverse, mesh.vertices[:, 2])
        x = (unique_xy[:, 0] - unique_xy[:, 0].min()) / np.ptp(unique_xy[:, 0])
        y = (unique_xy[:, 1] - unique_xy[:, 1].min()) / np.ptp(unique_xy[:, 1])
        center = (x >= .4) & (x <= .6) & (y >= .4) & (y <= .6)
        corners = ((x <= .1) | (x >= .9)) & ((y <= .1) | (y >= .9))
        center_z = float(np.median(top[center]))
        corner_z = float(np.median(top[corners]))
        # Use the exact source-image patches from the original bug diagnosis.
        # The lower image corners contain near objects, so broad four-corner
        # summaries above are retained only as supplementary measurements.
        columns = unique_xy[:, 0]
        rows = -unique_xy[:, 1]
        foreground = (columns >= 345) & (columns < 355) & (rows >= 258) & (rows < 268)
        background = (columns >= 0) & (columns < 10) & (rows >= 0) & (rows < 10)
        foreground_z = float(np.mean(top[foreground]))
        background_z = float(np.mean(top[background]))
        metrics[name] = {'path':str(paths[name]), 'sha256':digest(paths[name]),
            'vertices':len(mesh.vertices), 'faces':len(mesh.faces), 'same_xy_order_as_original_da3':same_xy,
            'exact_rgba_agreement_at_matching_xy':same_colors,
            'central_patch_median_front_z':center_z, 'corner_background_median_front_z':corner_z,
            'central_10x10_mean_front_z':foreground_z, 'top_left_10x10_mean_front_z':background_z,
            'foreground_above_background':foreground_z > background_z, 'foreground_minus_background':foreground_z-background_z}
    report = {'project':str(args.project), 'target_height_width':list(target), 'settings':settings,
        'model_info':model_info, 'elapsed_seconds':seconds, 'metrics':metrics,
        'originals_preserved':all(digest(p) == hashes_before[str(p)] for p in originals),
        'original_sha256':hashes_before,
        'region_definition':{'foreground_columns':[345,355], 'foreground_rows':[258,268], 'background_columns':[0,10], 'background_rows':[0,10], 'bounds':'start inclusive, end exclusive; source image coordinates', 'height':'mean of max Z at each XY'},
        'diagnostic_note':'An initial broad 40-60 percent center versus all four 10 percent corner comparison failed the direction oracle. The lower corners contain near objects. Those medians remain as supplementary metrics. The headline assertion uses the exact 10x10 foreground and top-left background patches from the original diagnosis.',
        'renders':{}}
    if args.render:
        try:
            import pyvista as pv
            state = project['session']['model_info']['accepted_mesh']['view_state']
            camera = state['camera']
            for name, mesh in meshes.items():
                faces = np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).reshape(-1)
                surface = pv.PolyData(mesh.vertices,faces)
                surface.point_data['RGBA'] = mesh.visual.vertex_colors
                plotter = pv.Plotter(off_screen=True,window_size=(1200,850))
                plotter.set_background(state['background'])
                plotter.add_mesh(surface,scalars='RGBA',rgba=True,show_scalar_bar=False,
                    smooth_shading=False, ambient=.2,diffuse=.8,specular=0)
                plotter.camera.position=camera['position']
                plotter.camera.focal_point=camera['focal_point']
                plotter.camera.up=camera['view_up']
                plotter.camera.view_angle=camera['view_angle']
                plotter.camera.parallel_scale=camera['parallel_scale']
                plotter.camera.parallel_projection=state['projection']=='Parallel'
                plotter.reset_camera_clipping_range()
                filename=args.output / f'{name}.png'
                plotter.screenshot(str(filename))
                plotter.close()
                report['renders'][name]=str(filename)
        except Exception as error:
            report['render_error']=f'{type(error).__name__}: {error}'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    assert report['originals_preserved']
    assert all(m['exact_rgba_agreement_at_matching_xy'] for m in metrics.values())
    assert not metrics['original_da3']['foreground_above_background']
    assert metrics['corrected_da3']['foreground_above_background']


if __name__ == '__main__':
    main()