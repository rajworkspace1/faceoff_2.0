document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.getElementById('navToggle');
    const sideNav = document.querySelector('.side-nav');
    const navLinks = document.querySelectorAll('.nav-link');
    const container = document.querySelector('.container');
    const overlay = document.createElement('div');
    overlay.className = 'nav-overlay';
    document.body.appendChild(overlay);

    const closeNav = () => {
        sideNav?.classList.remove('active');
        navToggle?.classList.remove('active');
        overlay.classList.remove('active');
        document.body.classList.remove('nav-open');
    };

    const toggleNav = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const isOpen = sideNav?.classList.contains('active');
        if (isOpen) {
            closeNav();
        } else {
            sideNav?.classList.add('active');
            navToggle?.classList.add('active');
            overlay.classList.add('active');
            document.body.classList.add('nav-open');
        }
    };

    // Add click handlers
    if (navToggle) {
        navToggle.addEventListener('click', toggleNav);
    }

    // Close nav when clicking overlay
    overlay.addEventListener('click', closeNav);

    // Close nav when clicking outside
    document.addEventListener('click', (e) => {
        if (!sideNav?.contains(e.target) &&
            !navToggle?.contains(e.target) &&
            sideNav?.classList.contains('active')) {
            closeNav();
        }
    });

    // Close nav when clicking links on mobile
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth < 992) {
                closeNav();
            }
        });
    });

    // Update active link when sections become visible
    const updateActiveLink = () => {
        navLinks.forEach(link => {
            const sectionId = link.getAttribute('data-section');
            const section = document.getElementById(sectionId);
            if (section?.style.display !== 'none') {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    };

    // Hook into existing section display logic
    const observer = new MutationObserver(updateActiveLink);
    document.querySelectorAll('[id$="-section"]').forEach(section => {
        observer.observe(section, { attributes: true });
    });

    // Smooth scroll to section when clicking nav links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('data-section');
            const section = document.getElementById(sectionId);
            if (section) {
                closeNav(); // Always close nav when clicking a link
                setTimeout(() => {
                    section.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }, 300); // Small delay to allow nav closing animation
            }
        });
    });

    // Update the process selected module handler
    const processSelectedModuleButton = document.getElementById('processSelectedModule');
    if (processSelectedModuleButton) {
        processSelectedModuleButton.addEventListener('click', async (event) => {
            const form = document.getElementById('uploadForm');
            const formData = new FormData(form);

            // First upload the video
            try {
                await uploadVideo(formData);
                // After successful upload, process the selected module
                startProcessing(event);
            } catch (error) {
                console.error('Error uploading video:', error);
                const statusEl = document.getElementById('status');
                if (statusEl) statusEl.innerText = `Error: ${error.message}`;
            }
        });
    }

    // Add module selection handling with validation
    const moduleItems = document.querySelectorAll('.module-item');
    let selectedModule = null;

    // Initially disable all module items
    moduleItems.forEach(item => {
        item.classList.add('disabled');
        item.style.pointerEvents = 'none';
        item.style.opacity = '0.6';
    });

    // Add file input change listener
    const fileInput = document.querySelector('input[type="file"]');
    fileInput.addEventListener('change', (e) => {
        const hasVideo = e.target.files.length > 0;
        moduleItems.forEach(item => {
            if (hasVideo) {
                item.classList.remove('disabled');
                item.style.pointerEvents = 'auto';
                item.style.opacity = '1';
            } else {
                item.classList.add('disabled');
                item.style.pointerEvents = 'none';
                item.style.opacity = '0.6';
            }
        });
    });

    moduleItems.forEach(item => {
        item.addEventListener('click', () => {
            const form = document.getElementById('uploadForm');
            if (!form.querySelector('input[type="file"]').files.length > 0) {
                alert('Please upload a video first!');
                return;
            }

            // Remove previous selection
            moduleItems.forEach(i => i.classList.remove('selected'));
            // Add selection to clicked item
            item.classList.add('selected');
            selectedModule = item.dataset.value;

            const event = {
                preventDefault: () => { },
                target: {
                    tagName: 'DIV',
                    dataset: { value: selectedModule }
                }
            };
            startProcessing(event);
            const modal = bootstrap.Modal.getInstance(document.getElementById('moduleSelectModal'));
            if (modal) {
                modal.hide();
            }
        });
    });
});

function getCSRFToken() {
    const csrfInput = document.querySelector('[name=csrfmiddlewaretoken]');
    return csrfInput ? csrfInput.value : '';
}


async function uploadVideo(formData) {
    try {
        const response = await fetch('/upload-video/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        return true;
    } catch (error) {
        console.error('Error:', error);
        throw new Error('Failed to upload video. Please try again.');
    }
}

async function processHR(formData) {

    try {
        const response = await fetch('/process-hr/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);

        }

        // Show HR section directly
        const hrSection = document.getElementById('hr-section');
        if (hrSection) {
            hrSection.style.display = 'block';
        }

        document.getElementById("HR-result").innerHTML = `
            <div class="results-container">
                <div class="overall-metrics">
                    <h3 class="content-heading">Detected Emotion</h3>

                    <div class="emotion-badge">
                        ${data.dominant_emotion}
                    </div>
                    <p>Average Heart Rate:</p>

                    <p>${data.avg_hr.toFixed(2)} BPM</p>
                
                </div>
                
            </div>
        `;
        // Removed matchContainerHeights();
        // if (!data.error) {
        //     trustScores.heart_rate = data.trust_score;
        // }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}

async function processHRV(formData) {

    try {
        const response = await fetch('/process-hrv/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);

        }

        // Show HRV section directly - Changed from hr-section to hrv-section
        const hrvSection = document.getElementById('hrv-section');
        if (hrvSection) {
            hrvSection.style.display = 'block';
        }

        document.getElementById("hrv-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Average Emotion</h3>
                        
                        <div class="emotion-badge">
                        ${data.avg_emotion}
                        
                        </div>

                        <p>SPO2:</p>
                        <p>${data.average_spo2.toFixed(2)} %</p>
                        <p>Mean Heart Rate:</p>

                        <p>${data.mean_hr.toFixed(2)} BPM</p>
                        <p>Standard Deviation of Normal-to-Normal intervals</p>

                        <p>${data.sdnn.toFixed(2)} ms</p>
                       
                    </div>

                    <div class="video-container">
                        <h3 class="content-heading">Analyzed Video</h3>
                        <video id="analyzedVideo1" controls autoplay muted loop src="${data.video_path}" ></video>
                    </div>
                    
                </div>
            `;

    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}

async function processVideoV1(formData) {

    try {
        const response = await fetch('/process_video_v1/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const videoV1Section = document.getElementById('video-v1-section');
        if (videoV1Section) {
            videoV1Section.style.display = 'block';
        }



        let segmentHTML = "";

        data.emotion_distribution.forEach(seg => {
            segmentHTML += `
                <div class="segment-item">
                    ${seg}
                </div>
            `;
        });

        document.getElementById("video-v1-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Dominant Emotion</h3>

                        <div class="emotion-badge">
                        ${data.dominant_emotion}
                        </div>
                        <div>
                        <p>Emotion Distribution</p>
                        ${segmentHTML}
                        </div>
                    </div>

                    <div class="video-container">
                        <h3 class="content-heading">Analyzed Video</h3>
                        <video id="analyzedVideo1" controls autoplay muted loop src="${data.video_path}" ></video>
                    </div>

                    
                </div>
            `;
        // Removed matchContainerHeights();
        // if (data.trust_score) {
        //     trustScores.video_v1 = data.trust_score;
        // }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}

async function processVideoV2(formData) {

    try {
        const response = await fetch('/process_video_v2/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const videoV1Section = document.getElementById('video-v2-section');
        if (videoV1Section) {
            videoV1Section.style.display = 'block';
        }



        document.getElementById("video-v2-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Major Emotion</h3>

                        <div class="emotion-badge">
                        ${data.major_emotion}
                        </div>
                       <p>HBA Fitness: ${data.hba_fitness}</p>
                    </div>
                    
                    
                </div>
            `;
        // if (!data.error) {
        //     accuracyScores.video_v2 = data.emotion_confidence;
        //     trustScores.video_v2 = data.trust_score;
        // }
        // Removed matchContainerHeights();
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}

async function processEye(formData) {

    try {
        const response = await fetch('/process_eye/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const eyeSection = document.getElementById('eye-section');
        if (eyeSection) {
            eyeSection.style.display = 'block';
        }



        document.getElementById("eye-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Detected Emotion</h3>
                        <p> overall_emotion : ${data.overall_emotion}</p>
                        <p> overall_eye_state : ${data.overall_eye_state}</p>
                        <p> baseline_state : ${data.baseline_state}</p>
                    </div>
                    <div class="video-container">
                        <h3 class="content-heading">Analyzed Video</h3>
                        <video id="analyzedVideo1" controls autoplay muted loop src="${data.output_video_url}" ></video>
                    </div>

                </div>
            `;
        // if (!data.error) {
        //     trustScores.eye = data.trust_score;
        // 
        // if (!data.error) {
        //     accuracyScores.video_v2 = data.emotion_confidence;
        // }
        // Removed matchContainerHeights();
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}


async function processMergedVideo(formData) {

    try {
        const response = await fetch('/merged_video/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const mergeSection = document.getElementById("merged-section");
        if (mergeSection) {
            mergeSection.style.display = 'block';
        }

        document.getElementById("merged-result").innerHTML = `
                <div class="results-container">
                    <div class="video-container">
                        <video id="analyzedVideo1" controls autoplay muted loop src="${data.video_path}" ></video>
                    </div>
                </div>
            `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}

async function processDeepfake(formData) {
    try {
        const response = await fetch('/process_deepfake/', {  // Changed to match urls.py pattern
            method: 'POST',
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const deepfakeSection = document.getElementById('deepfake-section');
        if (deepfakeSection) {
            deepfakeSection.style.display = 'block';
        }


        document.getElementById("deepfake-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Final Result</h3>
                        
                            <p>Audio: &nbsp ${data.audio}</p>
                            
                        
                            <p>Visual: &nbsp ${data.visual}</p>
                            

                            <p>Lip Sync: &nbsp ${data.lip_sync}</p>
                    </div>
                    <div class="video-container">
                        <h3 class="content-heading">Analyzed Video</h3>
                        <video id="analyzedVideo1" controls autoplay muted loop src="${data.video_path}" ></video>
                    </div>
                </div>
            `;
        // Removed matchContainerHeights();
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }

}


async function processAudioTone_V2(formData) {
    try {
        const response = await fetch('/process_audio_tone_V2/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const audioSection = document.getElementById('audio-section');
        if (audioSection) {
            audioSection.style.display = 'block';
        }


        let segmentHTML = "";

        data.segment_predictions.forEach(seg => {
            segmentHTML += `
                <div class="segment-item">
                    <strong>Segment ${seg.segment}</strong>: 
                    ${seg.emotion} 
                    <span class="confidence">(${(seg.confidence * 100).toFixed(1)}%)</span>
                </div>
            `;
        });

        document.getElementById("audio-tone-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Audio Tone(rudra)</h3>

                        <div class="emotion-badge">
                        ${data.overall_emotion}
                        </div>
                        <div class="segment-results">
                            <h4>Segment Details:</h4>
                            ${segmentHTML}
                        </div>
                        
                    </div>
                    
                </div>
            `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }
}

async function processAudioTone_V3(formData) {
    try {
        const response = await fetch('/process_audio_tone_V3/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const audioSection = document.getElementById('audio-section');
        if (audioSection) {
            audioSection.style.display = 'block';
        }


        document.getElementById("audio-tone-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Audio Tone(souradeep)</h3>

                        <div class="emotion-badge">
                        ${data.predicted_emotion}
                        </div>
                        <p> ${data.confidence}</p>
                        
                    </div>
                    
                </div>
            `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("status").innerText = `Error: ${error.message}`;
        throw error;
    }
}

async function processSpeech(formData) {
    try {
        const response = await fetch('/process_speech/', {
            method: 'POST',

            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const speechSection = document.getElementById('speech-section');
        if (speechSection) {
            speechSection.style.display = 'block';
        }

        // Check if there's an error (no audio detected)
        if (data.error) {
            document.getElementById("speech-result").innerHTML = `
                    <div class="results-container">
                        <div class="alert alert-warning">
                            <h3 class="content-heading">Speech Analysis Error</h3>
                            <p>${data.message || data.error}</p>
                        </div>
                        <div class="overall-metrics">
                            <h3 class="content-heading">Speech Tone</h3>
                            <div class="emotion-badge">
                                ${data.major_sentiment || 'Neutral'}
                            </div>
                            
                        </div>
                    </div>
                `;

            // Removed matchContainerHeights();
            return;
        }

        document.getElementById("speech-result").innerHTML = `
                <div class="results-container">
                    <div class="overall-metrics">
                        <h3 class="content-heading">Speech Sentiment</h3>

                        <p>sentiment: ${data.sentiment}</p>
                        <p>confidence: ${data.confidence}</p>
                        <p>language: ${data.language}</p>
                        
                    </div>
                     
                </div>
            `;
        // if (!data.error) {
        //     accuracyScores.speech = data.confidence;
        //     trustScores.speech = data.trust_score;
        // }
        // Removed matchContainerHeights();
    } catch (error) {
        console.error('Error:', error);
        document.getElementById("speech-result").innerHTML = `
            <div class="results-container">
                <div class="alert alert-danger">
                    <h3 class="content-heading">Processing Error</h3>
                    <p>${error.message}</p>
                </div>
            </div>
        `;

    }
}

function updateProgress(percent, status) {
    const progressBar = document.getElementById('progressBar');
    const progressBarInner = progressBar.querySelector('.progress-bar');
    const progressText = progressBar.querySelector('.progress-text');
    const statusEl = document.getElementById('status');

    if (progressBar) {
        progressBar.style.display = 'block';

        if (progressBar.classList.contains('circular')) {
            // Circular progress bar handling
            if (progressText) progressText.style.display = 'none';
            if (progressBarInner) {
                progressBarInner.style.width = '100%';
                progressBarInner.classList.add('spinner');
            }
        } else {
            // Horizontal progress bar handling
            progressBar.classList.remove('circular');
            if (progressBarInner) {
                progressBarInner.classList.remove('spinner');
                progressBarInner.style.transition = 'width 0.3s ease-in-out';
                progressBarInner.style.width = `${percent}%`;
                progressBarInner.setAttribute('aria-valuenow', percent);
            }
            if (progressText) {
                progressText.style.display = 'block';
                progressText.textContent = `${Math.round(percent)}%`;
            }
        }
    }

    if (statusEl) statusEl.innerText = status;

    // Update upload section state
    const uploadSection = document.querySelector('.upload-section');
    if (uploadSection) {
        if (percent > 0 && percent < 100) {
            uploadSection.classList.add('processing');
        } else {
            uploadSection.classList.remove('processing');
        }
    }
}

let sectionOrder = 0;

function showSection(sectionId) {
    const section = document.getElementById(sectionId);
    const grid = document.querySelector('.analysis-grid');

    if (section && grid) {
        // Get all visible sections
        const visibleSections = Array.from(grid.children).filter(
            s => s.style.display === 'block'
        );

        // Add slide down animation to existing visible sections with enhanced stagger
        visibleSections.forEach((sect, index) => {
            sect.style.transition = 'none'; // Reset transition
            sect.classList.remove('section-enter');

            // Force reflow
            void sect.offsetWidth;

            // Restore transition and add animation with stagger
            sect.style.transition = '';
            setTimeout(() => {
                sect.classList.add('section-slide-down');
            }, index * 80); // Increased stagger time
        });

        // Prepare new section
        section.style.opacity = '0';
        section.style.display = 'block';
        section.style.transform = 'translateY(-60px) scale(0.96) rotateX(15deg)';
        section.style.filter = 'blur(10px)';

        // Move new section to top
        grid.insertBefore(section, grid.firstChild);

        // Trigger animation with slight delay
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                section.classList.add('section-enter');
                section.style.opacity = '1';
                section.style.transform = '';
                section.style.filter = '';
            });
        });

        // Smooth scroll with enhanced easing
        const scrollTarget = Math.max(0, grid.offsetTop - 20);
        const startPosition = window.pageYOffset;
        const distance = scrollTarget - startPosition;

        const duration = 800;
        const start = performance.now();

        function scrollStep(currentTime) {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);

            // Custom easing function for smoother scroll
            const easeOutBack = t => {
                const c1 = 1.70158;
                const c3 = c1 + 1;
                return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
            };

            const easeProgress = easeOutBack(progress);
            window.scrollTo(0, startPosition + distance * easeProgress);

            if (progress < 1) {
                requestAnimationFrame(scrollStep);
            }
        }

        requestAnimationFrame(scrollStep);
    }
}



// Update the startProcessing function
async function startProcessing(event) {
    event.preventDefault();

    // Prevent multiple submissions
    let submitButton;
    if (event.target.tagName === 'FORM') {
        submitButton = event.target.querySelector('button[type="submit"]');
    } else {
        submitButton = document.getElementById('processSelectedModule');
    }
    if (submitButton?.disabled) return;
    if (submitButton) submitButton.disabled = true;

    const form = document.getElementById('uploadForm');
    const progressBar = document.getElementById('progressBar');
    const formData = new FormData(form);

    // Determine if processing a single module or all
    let selectedModule = null;
    if (event.target.tagName === 'FORM') {
        // Full analysis
        console.log("Full Analysis");
    } else {
        // Single module processing
        selectedModule = event.target.dataset?.value;
        console.log("Single Module Processing:", selectedModule);
    }



    // Reset section order counter
    sectionOrder = 0;

    // Hide all sections initially
    document.querySelectorAll('.analysis-grid > div').forEach(section => {
        section.style.display = 'none';
        section.style.order = '999';
        section.classList.remove('section-enter');
    });

    try {
        // Reset previous results
        const hrResult = document.getElementById('HR-result');
        const hrvResult = document.getElementById('hrv-result');
        const audioResult = document.getElementById('audio-tone-result');
        const deepfakeResult = document.getElementById('deepfake-result');
        const videoV1Result = document.getElementById('video-v1-result');
        const speechResult = document.getElementById('speech-result');
        const videoV2Result = document.getElementById('video-v2-result');
        const eyeResult = document.getElementById('eye-result');
        const mergedResult = document.getElementById('merged-result');


        if (hrResult) hrResult.innerHTML = '';
        if (hrvResult) hrvResult.innerHTML = '';
        if (audioResult) audioResult.innerHTML = '';
        if (deepfakeResult) deepfakeResult.innerHTML = '';
        if (videoV1Result) videoV1Result.innerHTML = '';
        if (videoV2Result) videoV2Result.innerHTML = '';
        if (speechResult) speechResult.innerHTML = '';
        if (eyeResult) eyeResult.innerHTML = '';
        if (mergedResult) mergedResult.innerHTML = '';


        // Hide analysis sections
        document.getElementById('hr-section').style.display = 'none';
        document.getElementById('hrv-section').style.display = 'none';
        document.getElementById('audio-section').style.display = 'none';
        document.getElementById('deepfake-section').style.display = 'none';
        document.getElementById('video-v1-section').style.display = 'none';
        document.getElementById('video-v2-section').style.display = 'none';
        document.getElementById('speech-section').style.display = 'none';
        document.getElementById('eye-section').style.display = 'none';
        document.getElementById('merged-section').style.display = 'none';


        // Show loading state
        if (progressBar) {
            progressBar.style.display = 'block';
            if (selectedModule) {
                progressBar.classList.add('circular');
            } else {
                progressBar.classList.remove('circular');
            }
        }

        // Step 1: Upload video (15%)
        updateProgress(10, "Uploading video...");
        await uploadVideo(formData);

        // Process single module
        if (selectedModule) {
            switch (selectedModule) {
                case 'deepfake':
                    updateProgress(20, "Analyzing deepfake...");
                    await processDeepfake(formData);
                    showSection('deepfake-section');
                    break;
                case 'video-v2':
                    updateProgress(40, "Analyzing Facial Expression...");
                    await processVideoV2(formData);
                    showSection('video-v2-section');
                    break;
                case 'eye':
                    updateProgress(60, "Analyzing Eye Movement...");
                    await processEye(formData);
                    showSection('eye-section');
                    break;
                case 'video-v1':
                    updateProgress(70, "Analyzing posture...");
                    await processVideoV1(formData);
                    showSection('video-v1-section');
                    break;
                case 'hr':
                    updateProgress(80, "Analyzing heart rate...");
                    await processHR(formData);
                    showSection('hr-section');
                    break;
                case 'hrv':
                    updateProgress(85, "Analyzing heart rate variability...");
                    await processHRV(formData);
                    showSection('hrv-section');
                    break;
                case 'speech':
                    updateProgress(90, "Analyzing speech sentiment...");
                    await processSpeech(formData);
                    showSection('speech-section');
                    break;
                case 'audio_V2':
                    updateProgress(95, "Analyzing voice emotions...");
                    await processAudioTone_V2(formData);
                    showSection('audio-section');
                    break;
                case 'audio_V3':
                    updateProgress(95, "Analyzing voice emotions...");
                    await processAudioTone_V3(formData);
                    showSection('audio-section');
                    break;
            }
            // Complete
            updateProgress(100, `Analysis complete!`);


        } else {
            // Full analysis flow
            // Step 2: Process deepfake (30%)
            updateProgress(40, "Analyzing deepfake...");
            await processDeepfake(formData);
            showSection('deepfake-section');


            //Step 3: Process video 2 (60%)
            updateProgress(50, "Analyzing Facial Expression...");
            await processVideoV2(formData);
            showSection('video-v2-section');

            // Step 4: Process eye(60%)
            updateProgress(60, "Analyzing Eye Movement...");
            await processEye(formData);
            showSection('eye-section');

            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for section to be visible

            // Step 5: Process video 1 (45%)
            updateProgress(70, "Analyzing posture...");
            await processVideoV1(formData);
            showSection('video-v1-section');

            // // Step 6: Process heart rate (75%)
            updateProgress(80, "Analyzing heart rate...");
            await processHR(formData);
            showSection('hr-section');

            // // Step 7: Process heart rate variability (80%)
            updateProgress(85, "Analyzing heart rate variability...");
            await processHRV(formData);
            showSection('hrv-section');

            // Step 7: Process speech sentiment (90%)
            updateProgress(90, "Analyzing speech sentiment...");
            await processSpeech(formData);
            showSection('speech-section');

            // Step 8: Process audio tone (100%)
            updateProgress(92, "Analyzing voice emotions...");
            await processAudioTone_V2(formData);
            showSection('audio-section');

            updateProgress(95, "merging the output videos...");
            await processMergedVideo(formData);
            showSection('merged-section');


            // Complete
            updateProgress(100, `Analysis complete!`);
        }


    } catch (error) {
        console.error('Error:', error);
        const statusEl = document.getElementById('status');
        if (statusEl) statusEl.innerText = `Error: ${error.message}`;
        updateProgress(0, `Error: ${error.message}`);
    } finally {
        // Hide loading state after 2 seconds if complete
        setTimeout(() => {
            if (submitButton) submitButton.disabled = false;
            if (progressBar && progressBar.querySelector('.progress-bar').style.width === '100%') {
                progressBar.style.display = 'none';
                progressBar.classList.remove('circular');
                document.querySelector('.upload-section').classList.remove('processing');
            }
        }, 2000);
    }
}
