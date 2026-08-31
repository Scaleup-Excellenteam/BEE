from pathlib import Path
from zipfile import ZipFile


def extract_archive(
    archive_path: str,
    destination_path: str,
) -> str:
    archive = Path(archive_path)
    destination = Path(destination_path)

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    destination.mkdir(parents=True, exist_ok=True)

    destination_resolved = destination.resolve()

    with ZipFile(archive, "r") as zip_file:
        for member in zip_file.infolist():
            target_path = (destination / member.filename).resolve()

            if (
                target_path != destination_resolved
                and destination_resolved not in target_path.parents
            ):
                raise ValueError(
                    f"Unsafe path inside ZIP: {member.filename}"
                )

        zip_file.extractall(destination)

    return str(destination)