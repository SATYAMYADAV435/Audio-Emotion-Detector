const API_URL = "http://localhost:8000";

// DOM Elements
const btnRecord = document.getElementById('btnRecord');
const fileInput = document.getElementById('audioUpload');
const systemLog = document.getElementById('systemLog');
const canvas = document.getElementById('visualizerCanvas');
const ctx = canvas.getContext('2d');
const emotionValue = document.getElementById('emotionValue');
const confidenceValue = document.getElementById('confidenceValue');
const pipelineStatus = document.getElementById('pipelineStatus');

// State
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let dataArray;

// Logger
function log(msg) {
    const div = document.createElement('div');
    div.innerText = `> ${msg}`;
    systemLog.appendChild(div);
    systemLog.scrollTop = systemLog.scrollHeight;
}

// Visualizer Setup
function setupVisualizer(stream) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    // Resume context if suspended
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    drawVisualizer();
}

function drawVisualizer() {
    if (!analyser) return;

    requestAnimationFrame(drawVisualizer);
    analyser.getByteFrequencyData(dataArray);

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    const barWidth = (width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        barHeight = dataArray[i] / 2;

        // Cyberpunk colors
        ctx.fillStyle = `rgb(${barHeight + 100}, 50, 255)`;
        ctx.fillRect(x, height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
    }
}

// Audio Recording Logic
btnRecord.addEventListener('click', async () => {
    if (!isRecording) {
        // Start Recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setupVisualizer(stream);

            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            audioChunks = [];
            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                await sendAudioForInference(audioBlob, "recorded_audio.wav");

                // Stop tracks
                stream.getTracks().forEach(track => track.stop());
            };

            isRecording = true;
            btnRecord.innerText = "🛑 Stop Recording";
            btnRecord.style.borderColor = "#FF0000";
            btnRecord.style.color = "#FF0000";
            pipelineStatus.innerText = "RECORDING...";
            log("Recording started...");

        } catch (err) {
            console.error(err);
            log("Error accessing microphone.");
        }
    } else {
        // Stop Recording
        mediaRecorder.stop();
        isRecording = false;
        btnRecord.innerText = "🎙️ Start Recording";
        btnRecord.style.borderColor = "#00FFFF";
        btnRecord.style.color = "#00FFFF";
        pipelineStatus.innerText = "PROCESSING...";
        log("Recording stopped. Sending to backend...");
    }
});

// File Upload Logic
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        log(`File selected: ${file.name}`);
        pipelineStatus.innerText = "UPLOADING...";

        // Create an audio context to visualize file?
        // Simpler for now: Just send it
        await sendAudioForInference(file, file.name);
    }
});

async function sendAudioForInference(blob, filename) {
    const formData = new FormData();
    formData.append("file", blob, filename);

    try {
        log("Sending to inference API...");
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const result = await response.json();

        // Update UI
        emotionValue.innerText = result.emotion.toUpperCase();
        confidenceValue.innerText = (result.confidence * 100).toFixed(1) + "%";

        pipelineStatus.innerText = "IDLE";
        log(`Success: Detected ${result.emotion}`);

    } catch (err) {
        log(`Inference Failed: ${err.message}`);
        pipelineStatus.innerText = "ERROR";
        emotionValue.innerText = "ERR";
        console.error(err);
    }
}

// Resize canvas
function resizeCanvas() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
