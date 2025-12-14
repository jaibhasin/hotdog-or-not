/**
 * Hot Dog or Not Hot Dog - Frontend Application
 * A tribute to Jian Yang's SeeFood app from HBO's Silicon Valley
 */

// API endpoint
const API_URL = '/predict';

// DOM Elements
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingSection = document.getElementById('loading-section');
const resultSection = document.getElementById('result-section');
const resultCard = document.getElementById('result-card');
const resultEmoji = document.getElementById('result-emoji');
const resultText = document.getElementById('result-text');
const resultConfidence = document.getElementById('result-confidence');
const tryAgainBtn = document.getElementById('try-again-btn');

// State
let selectedFile = null;

// ==================== Upload Handling ====================

// Click to upload
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// File selected via input
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// Handle selected file
function handleFile(file) {
    selectedFile = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        showSection('preview');
    };
    reader.readAsDataURL(file);
}

// ==================== Section Management ====================

function showSection(section) {
    // Hide all sections
    uploadZone.parentElement.classList.add('hidden');
    previewSection.classList.add('hidden');
    loadingSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    
    // Show requested section
    switch (section) {
        case 'upload':
            uploadZone.parentElement.classList.remove('hidden');
            break;
        case 'preview':
            previewSection.classList.remove('hidden');
            break;
        case 'loading':
            loadingSection.classList.remove('hidden');
            break;
        case 'result':
            resultSection.classList.remove('hidden');
            break;
    }
}

// ==================== Analyze Image ====================

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    showSection('loading');
    
    try {
        const result = await analyzeImage(selectedFile);
        displayResult(result);
    } catch (error) {
        console.error('Analysis failed:', error);
        displayError(error.message);
    }
});

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
    }
    
    return response.json();
}

// ==================== Display Results ====================

function displayResult(result) {
    const isHotDog = result.is_hotdog;
    const confidence = Math.round(result.confidence * 100);
    
    // Update result card
    resultCard.className = 'result-card ' + (isHotDog ? 'hotdog' : 'not-hotdog');
    resultEmoji.textContent = isHotDog ? '🌭' : '❌';
    resultText.textContent = result.prediction;
    resultConfidence.textContent = `${confidence}% confident`;
    
    showSection('result');
    
    // Log to console (for debugging)
    console.log('Prediction:', result);
}

function displayError(message) {
    resultCard.className = 'result-card not-hotdog';
    resultEmoji.textContent = '⚠️';
    resultText.textContent = 'Error';
    resultConfidence.textContent = message;
    
    showSection('result');
}

// ==================== Try Again ====================

tryAgainBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    showSection('upload');
});

// ==================== Initialize ====================

console.log('🌭 Hot Dog or Not Hot Dog');
console.log('A tribute to Jian Yang\'s SeeFood app');
