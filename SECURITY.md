# Security Policy: Holoscan Applications

## Reporting a Vulnerability

If you discover a potential security vulnerability, **do not open a public issue,
pull request, or discussion**.

- **Web (preferred):** Report it through the
  [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/).
- **Email:** Send details to [psirt@nvidia.com](mailto:psirt@nvidia.com). For
  encrypted communication, use the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub:** Use this repository's **Security** tab and select
  **Report a vulnerability** to submit a private report.

Include the project name and affected version or branch, vulnerability type,
reproduction steps, proof-of-concept code when available, and an impact
assessment. Detailed reports help NVIDIA evaluate and address issues faster.

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge the
report, validate and assess the issue, develop and test fixes, and publish a
security bulletin when appropriate.

## Supported Versions

Security fixes are applied to the current `main` branch. Application-specific
support depends on the Holoscan SDK versions declared in each application's
`metadata.json` and README.

## Security Architecture and Context

Holoscan Applications is a collection and build framework for accelerated
computing, streaming, and AI applications built with the NVIDIA Holoscan SDK.
The repository-level code discovers applications, registers CMake projects, and
uses the Holoscan CLI to build, run, test, and lint them. The current repository
contains the framework and application implementations under `applications/`.

This project operates at the application and build-tooling level. Its primary
security responsibility is to preserve the integrity of application source,
metadata, dependencies, container inputs, and CI automation. It does not provide
a shared authentication, authorization, network-security, or input-validation
layer for applications.

The principal security boundaries are:

- Application source, CMake logic, metadata, datasets, and configuration enter
  through repository contributions and application-specific setup.
- The Holoscan CLI, Holoscan SDK, Python package indexes, pre-commit hooks, GitHub
  Actions, and caller-provided container base image are external dependencies.
- Built applications may access GPUs, attached devices, displays, files,
  datasets, and external services with the permissions granted by the host or
  container runtime.
- CI executes repository tooling on GitHub-hosted runners. Local builds execute
  contributed build logic with the developer's privileges.

### Threat Model

1. **Malicious or vulnerable application contributions:** Source files, custom
   operators, CMake logic, and metadata added under `applications/` are executed
   during build, test, or runtime and could access host files, devices, GPU
   memory, credentials, or networks available to those processes.
2. **Build dependency substitution:** `Dockerfile` accepts a caller-provided
   `BASE_IMAGE`, and Python installation uses the default package index
   (PyPI). An untrusted image, compromised index, or package-name
   collision could introduce code during container or CI builds.
3. **CI and lint supply-chain compromise:** GitHub Actions and remote pre-commit
   hooks execute third-party code selected by version tags. A compromised
   upstream release or mutable tag could affect CI runners and developer
   workstations.
4. **Unsafe application-specific inputs:** Applications may consume
   datasets, device streams, model files, or external-service responses. The
   repository framework does not centrally authenticate, validate, or constrain
   those inputs, so malformed or adversarial data must be handled by each
   application and its dependencies. Some applications' model files come from
   third-party exporters that users install from PyPI outside the repository's
   pins, such as the Ultralytics exporter for `yolo26_detection`; their
   READMEs describe the risk and ask for an isolated environment.
5. **Resource exhaustion and unsafe device access:** Applications can perform
   GPU-intensive processing and interact with attached hardware. Unbounded
   inputs or incorrect device configuration could exhaust CPU, GPU, memory, or
   storage resources, or interfere with other workloads sharing the host.

### Critical Security Assumptions

- Maintainers review contributed application code, build logic, metadata, and
  dependency changes before execution or release.
- `BASE_IMAGE` is supplied from an approved registry and pinned to a trusted,
  preferably immutable digest.
- Python indexes, the Holoscan CLI and SDK, GitHub Actions, and pre-commit hook
  repositories are trusted and retrieved over authenticated TLS.
- The host OS, container runtime, GPU driver, Holoscan SDK, and hardware enforce
  process, device, and memory isolation appropriate to the deployment.
- Each application validates its own files, datasets, network messages, device
  data, model artifacts, and configuration before use.
- Secrets are supplied through deployment-specific secret management and are
  not committed to the repository, embedded in metadata, or copied into build
  contexts.
- Operators grant applications only the filesystem, network, GPU, display, and
  device permissions required for their documented operation.

## Dependency and Distribution Security

New or upgraded third-party dependencies must have their license and
redistribution terms reviewed before integration. Dependencies distributed in
source, binary, or container form must follow the applicable NVIDIA Open Source
Review Board process and preserve required license notices and corresponding
source. Release builds should use approved, pinned artifacts and retain a
reviewable dependency inventory.
