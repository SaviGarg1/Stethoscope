import argparse
import os
import uuid
from urllib import request as urllib_request


def build_multipart_body(field_name, filename, file_bytes, boundary):
    lines = []
    lines.append(f"--{boundary}")
    lines.append(f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"")
    lines.append("Content-Type: audio/wav")
    lines.append("")
    body = "\r\n".join(lines).encode("utf-8") + b"\r\n" + file_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body


def upload_wav_file(filepath, endpoint="http://127.0.0.1:5000/arduino/upload"):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    with open(filepath, "rb") as fh:
        file_bytes = fh.read()

    boundary = uuid.uuid4().hex
    body = build_multipart_body("audio_file", os.path.basename(filepath), file_bytes, boundary)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    req = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=20) as resp:
        status_code = resp.getcode()
        response_body = resp.read().decode("utf-8", errors="replace")
    return status_code, response_body


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a WAV file to the Flask Arduino upload endpoint.")
    parser.add_argument("file", help="Path to the WAV file to upload.")
    parser.add_argument("--url", default="http://127.0.0.1:5000/arduino/upload", help="Flask endpoint URL.")
    args = parser.parse_args()

    try:
        code, body = upload_wav_file(args.file, args.url)
        print(f"Uploaded {args.file} -> {args.url} (status {code})")
        print(body)
    except Exception as exc:
        print(f"Upload failed: {exc}")
