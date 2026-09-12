# Model licenses and provenance

Official sources checked 2026-09-11 07:09, America/Chicago. These are general
guidelines, not legal advice. EdgeMesh being free and non-commercial does not
change the terms attached to model weights. The labels record which model was
used; they do not assign a legal classification to every generated output.

## Checkpoint-specific guidance

| Model weights | Terms | Practical guidance |
| --- | --- | --- |
| Depth Anything V1 Large (diagnostic comparison) | Apache 2.0 | Commercial and non-commercial use permitted under the license conditions. The comparison adapter is not listed in the depth-generation dropdown. |
| Depth Anything V2 Large (current loader), Base | CC BY-NC 4.0 | Non-commercial use; preserve attribution and license notices for covered material. Obtain separate permission before commercial use. |
| Depth Anything V2 Small | Apache 2.0 | Commercial and non-commercial use permitted under the license conditions. |
| Depth Pro | Apple Machine Learning Research Model license | Non-commercial scientific research and academic development only. The license excludes product development and commercial exploitation. Free creative/product use is not automatically covered. |
| SAM 2.1 Tiny | Apache 2.0 | Preserve required notices and other license conditions. |
| Locally registered MiDaS / DPT weights | Unverified checkpoint terms | The source repository's MIT license does not establish the terms of arbitrary user-supplied weights. |

Sources: [Depth Anything V2 checkpoint licenses](https://github.com/DepthAnything/Depth-Anything-V2#license),
[V1 Large checkpoint card](https://huggingface.co/LiheYoung/depth-anything-large-hf),
[exact V2 Large Transformers card](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf),
[Apple weight license](https://huggingface.co/apple/DepthPro/blob/main/LICENSE),
[Depth Pro Transformers card](https://huggingface.co/apple/DepthPro-hf),
[SAM 2 weight licenses](https://github.com/facebookresearch/sam2#license),
[MiDaS source license](https://github.com/isl-org/MiDaS/blob/master/LICENSE).
The [CC BY-NC summary](https://creativecommons.org/licenses/by-nc/4.0/) explains
attribution and non-commercial conditions; a user's or organization's nonprofit
status alone does not decide whether a particular use is commercial.

[Depth Anything 3](https://depth-anything-3.github.io/) also has mixed terms:
Small, Base, Mono-Large, and Metric-Large use Apache 2.0. Large, Giant, and Nested
Giant-Large use CC BY-NC 4.0, including refreshed `-1.1` checkpoints. The catalog
records these identities for accurate provenance; catalog membership alone does
not mean an inference adapter is installed. The [official model table](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards)
is the source for these distinctions.

No separate paid commercial-license offer was verified in the reviewed official
DA2, DA3, or Depth Pro sources. EdgeMesh therefore says that no offer is published
in those sources and links to the publisher, rather than claiming a purchase route
exists. Apache-covered variants include commercial permissions under their terms.
Future verified commercial offers can be added through the catalog's explicit
`commercial_status`, `commercial_info`, and `commercial_url` fields.

## Download acknowledgment

Restricted model preparation requires both download permission and an explicit
model-license acknowledgment before the processor or weights are downloaded.
The accepted record is scoped to the exact model ID, immutable revision, and
catalog policy digest. A new revision or policy prompts again. Checking the
download box alone is insufficient. A cached offline preparation needs no new
download acknowledgment and still carries its NC or research-only label.

The GUI preflight reads local state only. If no immutable revision is known yet,
an explicit acknowledgment is retained for one preparation in the current
process. That preparation resolves an immutable revision using Hub metadata and
persists acceptance against that revision before downloading the model files.
Declining, closing the dialog, or returning a truthy value other than literal
`True` from the injected callback does not record consent.

## Binding the result to its models

The plaintext project record contains model IDs, revisions, available weight-file
SHA-256 hashes, license policies, parameters, and mesh provenance. Known catalog
IDs always determine their license policy; editable project claims cannot turn
a known NC checkpoint into a permissive model. Unknown explicit checkpoint IDs
remain unverified even if they name a familiar model type.

The provenance seal hashes exact ordered vertices and triangle indices using
fixed little-endian binary arrays with framed dimensions. An HMAC-SHA256 binds
that geometry hash to the actual depth and mask model identities and parameters.
Cached weight hashes are captured when checkpoint files are available. They are
local identity evidence, not an independent certificate from the model publisher.
An imported mask with unverified model ancestry stays unverified even when a new
local signature protects later edits or newly generated depth geometry. A valid
local signature alone does not establish the source of an imported asset.

The local signing key stays outside the project. Windows encrypts it with
current-user DPAPI; other platforms use a private directory and mode-0600 file.
Completed keys are published without overwriting an existing profile key, so
simultaneous app launches share one key. The key is never exported with a mesh.

`verified` means the signature matches this profile; geometry is checked only
when the actual arrays are supplied. `tampered` means metadata/signature or
geometry changed or mismatched; it does not establish intent. `unverified` means
the original profile key, supported metadata, or protection mechanism is absent.
Opening a project on another computer normally produces `unverified`. If local
key storage fails, generation retains unsigned metadata and logs the failure.

This is tamper evidence against casual edits, not DRM, a watermark embedded in
the shape, or a tamperproof guarantee. An owner who controls the code or local
profile can replace the key and re-sign data. Removing a sidecar or converting a
mesh can remove provenance; the app must present that as missing/unverified.
