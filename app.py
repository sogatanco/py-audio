import os
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import whisper
from werkzeug.utils import secure_filename
import threading
import uuid
from pyannote.audio import Pipeline
from dotenv import load_dotenv

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

load_dotenv()
model = whisper.load_model("small")

DIARIZATION_TOKEN = os.environ.get("HF_TOKEN")
pipeline = None
if DIARIZATION_TOKEN:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=DIARIZATION_TOKEN)

progress_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_async(task_id, filepath, result_filename, language):
    progress_status[task_id] = {'status': 'processing', 'progress': 10}
    try:
        transcribe_kwargs = {}
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        result = model.transcribe(filepath, **transcribe_kwargs)
        progress_status[task_id]['progress'] = 60
        text = result['text']

        diarization_result = None
        diarized_text = ""
        if pipeline:
            diarization_result = pipeline(filepath)
            segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                # Note: whisper.transcribe per segment is not directly supported, you may want to adjust this logic
                seg_text = text  # Simplified, consider segment transcription if needed
                segments.append(f"[{speaker}] {seg_text.strip()}")
            diarized_text = "\n".join(segments)
        else:
            diarized_text = text

        progress_status[task_id]['progress'] = 90
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(diarized_text)
        progress_status[task_id] = {
            'status': 'done',
            'progress': 100,
            'result_filename': result_filename,
            'text': diarized_text
        }
    except Exception as e:
        progress_status[task_id] = {'status': 'error', 'progress': 100, 'error': str(e)}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        language = request.form.get('language', 'auto')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_filename = filename + '.txt'
            task_id = str(uuid.uuid4())
            thread = threading.Thread(target=transcribe_async, args=(task_id, filepath, result_filename, language))
            thread.start()
            return redirect(url_for('processing', task_id=task_id))
        else:
            return render_template('index.html', error="File type not allowed")
    return render_template('index.html')

@app.route('/processing/<task_id>')
def processing(task_id):
    return render_template('processing.html', task_id=task_id)

@app.route('/progress/<task_id>')
def progress(task_id):
    status = progress_status.get(task_id, {'status': 'not_found', 'progress': 0})
    return jsonify(status)

@app.route('/result/<task_id>')
def result(task_id):
    status = progress_status.get(task_id)
    if not status or status.get('status') != 'done':
        return redirect(url_for('processing', task_id=task_id))
    return render_template('result.html', text=status['text'], download_url=url_for('download', filename=status['result_filename']))

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
