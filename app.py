"""
Web interface for invoice screenshot cropper.
Run: python3 app.py
Open: http://localhost:5000 (or from phone on same Wi-Fi)
"""

import io
import base64
import json
import zipfile
from pathlib import Path

from flask import Flask, request, render_template_string, send_file, session
from PIL import Image

from detectors import CardBoundaryDetector
from cropper import ImageCropper

app = Flask(__name__)
app.secret_key = "invoice-cropper"

detector = CardBoundaryDetector()
cropper = ImageCropper(detector=detector, padding=0)

# In-memory store for last batch of cropped images (single user)
last_batch = {}

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Invoice Cropper</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f5f7;
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
        }
        h1 { font-size: 24px; margin-bottom: 8px; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 24px; }
        .upload-area {
            background: white;
            border: 2px dashed #ccc;
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 16px;
        }
        input[type="file"] { display: none; }
        .btn {
            display: block;
            background: #007aff;
            color: white;
            padding: 14px 28px;
            border-radius: 12px;
            border: none;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 8px;
            text-align: center;
            text-decoration: none;
        }
        .btn:disabled { background: #ccc; }
        .btn-green { background: #34c759; margin-top: 16px; }
        .btn-reset { background: #ff3b30; margin-top: 12px; }
        .file-count {
            margin: 12px 0;
            font-size: 14px;
            color: #333;
        }
        .results { margin-top: 24px; }
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .result-card img {
            width: 100%;
            border-radius: 8px;
        }
        .result-card .filename {
            font-size: 13px;
            color: #666;
            margin-top: 8px;
        }
        .save-hint {
            background: #e8f5e9;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
            font-size: 14px;
            color: #2e7d32;
            text-align: center;
        }
        .save-steps {
            background: white;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
            font-size: 13px;
            color: #666;
            line-height: 1.6;
        }
        .spinner {
            display: none;
            margin: 20px auto;
            width: 40px;
            height: 40px;
            border: 4px solid #eee;
            border-top: 4px solid #007aff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        .spinner.active { display: block; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status { text-align: center; color: #666; font-size: 14px; margin-bottom: 12px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>Invoice Cropper</h1>
    <p class="subtitle">Select invoice screenshots to crop</p>

    <div id="uploadSection">
        <div class="upload-area" onclick="document.getElementById('files').click()">
            <p style="font-size: 18px; margin-bottom: 8px;">Tap to select screenshots</p>
            <p style="color: #999; font-size: 13px;">PNG or JPG</p>
        </div>
        <input type="file" id="files" multiple accept="image/*">
        <div class="file-count" id="fileCount"></div>
        <button class="btn" id="cropBtn" disabled onclick="cropImages()">Crop Images</button>
    </div>

    <div class="spinner" id="spinner"></div>
    <p class="status" id="status"></p>

    <div class="results" id="results">
        <div class="save-hint hidden" id="saveHint"></div>
        <a class="btn btn-green hidden" id="downloadBtn" href="/download-zip">Download All as ZIP</a>
        <div class="save-steps hidden" id="saveSteps">
            After downloading: Open <strong>Files</strong> app → find the ZIP →
            tap to unzip → select all images → tap <strong>Share</strong> → <strong>Save to Photos</strong>
        </div>
    </div>

    <button class="btn btn-reset hidden" id="resetBtn" onclick="resetForm()">Crop More</button>

    <script>
        const filesInput = document.getElementById('files');
        const fileCount = document.getElementById('fileCount');
        const cropBtn = document.getElementById('cropBtn');
        const spinner = document.getElementById('spinner');
        const status = document.getElementById('status');
        const results = document.getElementById('results');
        const saveHint = document.getElementById('saveHint');
        const saveSteps = document.getElementById('saveSteps');
        const downloadBtn = document.getElementById('downloadBtn');
        const uploadSection = document.getElementById('uploadSection');
        const resetBtn = document.getElementById('resetBtn');

        filesInput.addEventListener('change', () => {
            const n = filesInput.files.length;
            fileCount.textContent = n > 0 ? n + ' screenshot' + (n > 1 ? 's' : '') + ' selected' : '';
            cropBtn.disabled = n === 0;
        });

        async function cropImages() {
            cropBtn.disabled = true;
            spinner.classList.add('active');
            status.textContent = 'Cropping...';

            const formData = new FormData();
            for (const file of filesInput.files) {
                formData.append('files', file);
            }

            try {
                const resp = await fetch('/crop', { method: 'POST', body: formData });
                const data = await resp.json();

                if (data.error) {
                    status.textContent = 'Error: ' + data.error;
                    cropBtn.disabled = false;
                    spinner.classList.remove('active');
                    return;
                }

                const successCount = data.results.filter(r => r.success).length;
                status.textContent = successCount + ' image' + (successCount > 1 ? 's' : '') + ' cropped!';

                saveHint.textContent = successCount + ' cropped invoice' + (successCount > 1 ? 's' : '') + ' ready to download';
                saveHint.classList.remove('hidden');
                downloadBtn.classList.remove('hidden');
                saveSteps.classList.remove('hidden');
                uploadSection.classList.add('hidden');
                resetBtn.classList.remove('hidden');

                // Show previews
                for (const item of data.results) {
                    if (item.success) {
                        const card = document.createElement('div');
                        card.className = 'result-card';

                        const img = document.createElement('img');
                        img.src = 'data:image/jpeg;base64,' + item.image;
                        card.appendChild(img);

                        const name = document.createElement('p');
                        name.className = 'filename';
                        name.textContent = item.filename;
                        card.appendChild(name);

                        results.appendChild(card);
                    } else {
                        const card = document.createElement('div');
                        card.className = 'result-card';
                        card.innerHTML = '<p style="color:#ff3b30">Failed: ' + item.filename + ' — ' + item.error + '</p>';
                        results.appendChild(card);
                    }
                }
            } catch (err) {
                status.textContent = 'Error: ' + err.message;
            }

            spinner.classList.remove('active');
            cropBtn.disabled = false;
        }

        function resetForm() {
            const cards = results.querySelectorAll('.result-card');
            cards.forEach(c => c.remove());
            saveHint.classList.add('hidden');
            downloadBtn.classList.add('hidden');
            saveSteps.classList.add('hidden');
            resetBtn.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            filesInput.value = '';
            fileCount.textContent = '';
            cropBtn.disabled = true;
            status.textContent = '';
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE)


@app.route("/crop", methods=["POST"])
def crop():
    files = request.files.getlist("files")

    if not files:
        return json.dumps({"error": "No files uploaded"}), 400

    results = []
    last_batch.clear()

    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
            cropped = cropper.crop(img)

            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=95)
            image_bytes = buf.getvalue()

            b64 = base64.b64encode(image_bytes).decode("utf-8")

            name = Path(f.filename).stem + "_cropped.jpg"
            last_batch[name] = image_bytes

            results.append({
                "filename": name,
                "success": True,
                "image": b64,
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "success": False,
                "error": str(e),
            })

    return json.dumps({"results": results}), 200, {"Content-Type": "application/json"}


@app.route("/download-zip")
def download_zip():
    if not last_batch:
        return "No cropped images available", 404

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, image_bytes in last_batch.items():
            zf.writestr(name, image_bytes)

    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip", download_name="cropped_invoices.zip")


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5000

    print(f"\n  Invoice Cropper running!")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Phone:   http://{local_ip}:{port}")
    print(f"\n  (Make sure your phone is on the same Wi-Fi)\n")

    app.run(host="0.0.0.0", port=port, debug=True)
