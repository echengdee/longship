window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "https://storage.googleapis.com/nerfies-public/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Remove interactive demo section on mobile devices to prevent loading
    if (window.innerWidth <= 768) {
        $('.interactive-demo-section').remove();
        console.log('Interactive demo removed on mobile device');
        
        // Remove autoplay and loop from all videos on mobile (except flagship, roll, and climb_4)
        // Also add lazy loading to all videos
        $('video').each(function() {
            var videoSrc = $(this).attr('src') || $(this).find('source').attr('src') || '';
            var isFlagship = videoSrc.includes('flagship');
            var isRoll = videoSrc.includes('roll');
            var isBox1 = videoSrc.includes('box_1');
            
            if (!isFlagship && !isRoll && !isBox1) {
                $(this).removeAttr('autoplay');
                $(this).removeAttr('loop');
                $(this).attr('loading', 'lazy');
                // Pause any videos that may have already started
                this.pause();
            }
        });
        console.log('Video autoplay and loop removed on mobile device (except flagship, roll, and box_1)');
        console.log('Lazy loading added to all videos on mobile device');
    }

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 2,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class (including mobile carousels)
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    // Box Augmentation Demo functionality (only initialize if legacy demo elements exist)
    if (document.getElementById('pose-demo-iframe')) {
        initializeBoxAugmentationDemo();
    }

})

// Box Augmentation Demo Functions
function initializeBoxAugmentationDemo() {
    // Demo state
    let demoState = {
        currentPose: 'original',
        currentSize: 'original',
        currentTerrain: '1.2', // Default to highest terrain height
        currentEmbodiment: 't1_box',
        currentBaselineMethod: 'gmr',
        currentBaselineTask: 'box',
        currentOmniRetargetTask: 'box',
        currentLafan: 'dance1_subject1',
        isFullscreen: false,
        fullscreenType: null,
        fullscreenContainer: null
    };

    // Interactive demo configuration with correct file mappings
    const poseConfigs = {
        original: {
            html: 'sub3_largebox_003_original.html',
            title: 'Original Object Pose',
            description: 'Interactive 3D visualization with original object placement'
        },
        rot45cw: {
            html: 'sub3_largebox_003_rot_1.html',
            title: 'Rotated 45° Clockwise',
            description: 'Interactive 3D visualization with object rotated 45 degrees clockwise'
        },
        rot45ccw: {
            html: 'sub3_largebox_003_rot_0.html',
            title: 'Rotated 45° Counter-Clockwise',
            description: 'Interactive 3D visualization with object rotated 45 degrees counter-clockwise'
        },
        translate_front: {
            html: 'sub3_largebox_003_trans_0.html',
            title: 'Translated Forward',
            description: 'Interactive 3D visualization with object moved forward'
        },
        translate_left: {
            html: 'sub3_largebox_003_trans_1.html',
            title: 'Translated Left',
            description: 'Interactive 3D visualization with object moved to the left'
        },
        translate_right: {
            html: 'sub3_largebox_003_trans_2.html',
            title: 'Translated Right',
            description: 'Interactive 3D visualization with object moved to the right'
        }
    };

    const sizeConfigs = {
        original: {
            html: 'sub3_largebox_003_original.html',
            title: 'Original Object Size',
            description: 'Interactive 3D visualization with original object size'
        },
        small: {
            html: 'sub3_largebox_003_small.html',
            title: 'Small Object Size',
            description: 'Interactive 3D visualization with smaller object'
        },
        large: {
            html: 'sub3_largebox_003_large.html',
            title: 'Large Object Size',
            description: 'Interactive 3D visualization with larger object'
        }
    };

    const terrainConfigs = {
        '0.8': {
            html: 'climb_0.8.html',
            title: 'Terrain Height 0.8×',
            description: 'Interactive 3D visualization with 80% terrain height'
        },
        '0.9': {
            html: 'climb_0.9.html',
            title: 'Terrain Height 0.9×',
            description: 'Interactive 3D visualization with 90% terrain height'
        },
        original: {
            html: 'climb_1.0.html',
            title: 'Original Terrain Height',
            description: 'Interactive 3D visualization with original terrain height'
        },
        '1.1': {
            html: 'climb_1.1.html',
            title: 'Terrain Height 1.1×',
            description: 'Interactive 3D visualization with 110% terrain height'
        },
        '1.2': {
            html: 'climb_1.2.html',
            title: 'Terrain Height 1.2×',
            description: 'Interactive 3D visualization with 120% terrain height'
        }
    };

    const embodimentConfigs = {
        h1_box: {
            html: 'h1_box.html',
            title: 'H1 Robot - Box Task',
            description: 'Interactive 3D visualization with H1 robot carrying object'
        },
        h1_climb: {
            html: 'h1_climb.html',
            title: 'H1 Robot - Climbing Task',
            description: 'Interactive 3D visualization with H1 robot climbing'
        },
        t1_box: {
            html: 't1_box.html',
            title: 'T1 Robot - Box Task',
            description: 'Interactive 3D visualization with T1 robot carrying object'
        },
        t1_climb: {
            html: 't1_climb.html',
            title: 'T1 Robot - Climbing Task',
            description: 'Interactive 3D visualization with T1 robot climbing'
        }
    };

    const baselineConfigs = {
        gmr_box: {
            html: 'gmr_box.html',
            title: 'GMR - Box',
        },
        gmr_climb: {
            html: 'gmr_climb.html',
            title: 'GMR - Climb',
        },
        phc_box: {
            html: 'phc_box.html',
            title: 'PHC - Box',
        },
        phc_climb: {
            html: 'phc_climb.html',
            title: 'PHC - Climb',
        },
        omniretarget_box: {
            html: 'omniretarget_box.html',
            title: 'OmniRetarget - Box',
        },
        omniretarget_climb: {
            html: 'omniretarget_climb.html',
            title: 'OmniRetarget - Climb',
        }
    };

    const lafanConfigs = {
        dance1_subject1: {
            html: 'dance1_subject1.html',
            title: 'Dance 1 - Subject 1',
            description: 'LAFAN1 sequence: Dance 1, Subject 1'
        },
        ground1_subject4: {
            html: 'ground1_subject4.html',
            title: 'Ground 1 - Subject 4',
            description: 'LAFAN1 sequence: Ground 1, Subject 4'
        },
        multipleActions1_subject4: {
            html: 'multipleActions1_subject4.html',
            title: 'Multiple Actions 1 - Subject 4',
            description: 'LAFAN1 sequence: Multiple Actions 1, Subject 4'
        },
        obstacles1_subject1: {
            html: 'obstacles1_subject1.html',
            title: 'Obstacles 1 - Subject 1',
            description: 'LAFAN1 sequence: Obstacles 1, Subject 1'
        }
    };

    // File path configurations
    const filePaths = {
        pose: './static/interactive_demo/box_pose/',
        size: './static/interactive_demo/box_size/',
        terrain: './static/interactive_demo/terrain/',
        embodiment: './static/interactive_demo/embodiment/',
        baseline: './static/interactive_demo/baseline/',
        lafan: './static/interactive_demo/lafan/'
    };

    // Get DOM elements
    const poseIframe = document.getElementById('pose-demo-iframe');
    const sizeIframe = document.getElementById('size-demo-iframe');
    const terrainIframe = document.getElementById('terrain-demo-iframe');
    const embodimentIframe = document.getElementById('embodiment-demo-iframe');
    const baselineMethodIframe = document.getElementById('baseline-method-demo-iframe');
    const omniretargetIframe = document.getElementById('omniretarget-demo-iframe');
    const lafanIframe = document.getElementById('lafan-demo-iframe');
    const poseDescription = document.getElementById('pose-description');
    const sizeDescription = document.getElementById('size-description');
    const terrainDescription = document.getElementById('terrain-description');
    const embodimentDescription = document.getElementById('embodiment-description');
    const baselineMethodDescription = document.getElementById('baseline-method-description');
    const omniretargetDescription = document.getElementById('omniretarget-description');
    const baselineMethodTitle = document.getElementById('baseline-method-title');
    const omniretargetTitle = document.getElementById('omniretarget-title');
    const lafanDescription = document.getElementById('lafan-description');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const fullscreenPoseBtn = document.getElementById('fullscreen-pose-btn');
    const fullscreenSizeBtn = document.getElementById('fullscreen-size-btn');
    const fullscreenTerrainBtn = document.getElementById('fullscreen-terrain-btn');
    const fullscreenEmbodimentBtn = document.getElementById('fullscreen-embodiment-btn');
    const fullscreenBaselineBtn = document.getElementById('fullscreen-baseline-btn');
    const fullscreenLafanBtn = document.getElementById('fullscreen-lafan-btn');
    const sizeSelect = document.getElementById('size-select');
    const loadPoseDemoBtn = document.getElementById('load-pose-demo-btn');
    const loadSizeDemoBtn = document.getElementById('load-size-demo-btn');
    const loadTerrainDemoBtn = document.getElementById('load-terrain-demo-btn');
    const loadEmbodimentDemoBtn = document.getElementById('load-embodiment-demo-btn');
    const loadBaselineMethodDemoBtn = document.getElementById('load-baseline-method-demo-btn');
    const loadLafanDemoBtn = document.getElementById('load-lafan-demo-btn');
    const poseOptions = document.querySelectorAll('.pose-option');
    const sizeOptions = document.querySelectorAll('.size-option');
    const terrainOptions = document.querySelectorAll('.terrain-option');
    const embodimentOptions = document.querySelectorAll('.embodiment-option');
    const baselineMethodOptions = document.querySelectorAll('.baseline-method-option');
    const baselineTaskOptions = document.querySelectorAll('.baseline-task-option');
    const omniretargetTaskOptions = document.querySelectorAll('.omniretarget-task-option');
    const lafanOptions = document.querySelectorAll('.lafan-option');

    // Check if required elements exist
    if (!poseIframe) {
        console.error('poseIframe not found!');
        return;
    }
    if (poseOptions.length === 0) {
        console.error('No pose options found!');
        return;
    }
    if (sizeOptions.length === 0) {
        console.error('No size options found!');
        return;
    }

    // Event listeners
    loadPoseDemoBtn.addEventListener('click', loadPoseDemo);
    loadSizeDemoBtn.addEventListener('click', loadSizeDemo);
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    if (fullscreenPoseBtn) {
        fullscreenPoseBtn.addEventListener('click', () => toggleFullscreen('pose'));
    }
    if (fullscreenSizeBtn) {
        fullscreenSizeBtn.addEventListener('click', () => toggleFullscreen('size'));
    }
    if (loadTerrainDemoBtn) {
        loadTerrainDemoBtn.addEventListener('click', loadTerrainDemo);
    }
    if (loadEmbodimentDemoBtn) {
        loadEmbodimentDemoBtn.addEventListener('click', loadEmbodimentDemo);
    }
    if (loadBaselineMethodDemoBtn) {
        loadBaselineMethodDemoBtn.addEventListener('click', () => {
            loadBaselineMethodDemo();
            loadOmniRetargetDemo();
        });
    }
    if (loadLafanDemoBtn) {
        loadLafanDemoBtn.addEventListener('click', loadLafanDemo);
    }
    if (fullscreenTerrainBtn) {
        fullscreenTerrainBtn.addEventListener('click', () => toggleFullscreen('terrain'));
    }
    if (fullscreenEmbodimentBtn) {
        fullscreenEmbodimentBtn.addEventListener('click', () => toggleFullscreen('embodiment'));
    }
    const fullscreenBaselineMethodBtn = document.getElementById('fullscreen-baseline-method-btn');
    const fullscreenOmniRetargetBtn = document.getElementById('fullscreen-omniretarget-btn');
    if (fullscreenBaselineMethodBtn) {
        fullscreenBaselineMethodBtn.addEventListener('click', () => toggleFullscreen('baselineMethod'));
    }
    if (fullscreenOmniRetargetBtn) {
        fullscreenOmniRetargetBtn.addEventListener('click', () => toggleFullscreen('omniretarget'));
    }
    if (fullscreenLafanBtn) {
        fullscreenLafanBtn.addEventListener('click', () => toggleFullscreen('lafan'));
    }

    // Add click listeners to pose options
    poseOptions.forEach((option, index) => {
        // Primary click handler
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectPoseOption(option);
        });
        
        // Touch support for mobile devices
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectPoseOption(option);
        });
    });

    // Add click listeners to size options
    sizeOptions.forEach((option, index) => {
        // Primary click handler
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectSizeOption(option);
        });
        
        // Touch support for mobile devices
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectSizeOption(option);
        });
    });

    // Add click listeners to terrain options
    terrainOptions.forEach((option, index) => {
        // Primary click handler
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectTerrainOption(option);
        });
        
        // Touch support for mobile devices
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectTerrainOption(option);
        });
    });

    // Add click listeners to embodiment options
    embodimentOptions.forEach((option, index) => {
        // Primary click handler
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectEmbodimentOption(option);
        });
        
        // Touch support for mobile devices
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectEmbodimentOption(option);
        });
    });

    // Add click listeners to baseline method options
    baselineMethodOptions.forEach((option) => {
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectBaselineMethodOption(option);
        });
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectBaselineMethodOption(option);
        });
    });

    // Add click listeners to baseline task options
    baselineTaskOptions.forEach((option) => {
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectBaselineTaskOption(option);
        });
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectBaselineTaskOption(option);
        });
    });

    // Add click listeners to omniretarget task options
    omniretargetTaskOptions.forEach((option) => {
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectOmniRetargetTaskOption(option);
        });
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectOmniRetargetTaskOption(option);
        });
    });

    // Add click listeners to lafan options
    lafanOptions.forEach((option) => {
        option.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectLafanOption(option);
        });
        option.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            selectLafanOption(option);
        });
    });

    // Initialize with desired defaults
    const defaultPoseOption = Array.from(poseOptions).find(o => o.dataset.pose === 'rot45ccw') || poseOptions[0];
    selectPoseOption(defaultPoseOption);
    const defaultSizeOption = Array.from(sizeOptions).find(o => o.dataset.size === 'small') || sizeOptions[0];
    selectSizeOption(defaultSizeOption);
    selectTerrainOption(terrainOptions[4]); // Select "1.2" terrain by default (index 4)
    selectEmbodimentOption(embodimentOptions[0]); // Select first embodiment by default
    if (baselineMethodOptions.length > 0) selectBaselineMethodOption(baselineMethodOptions[0]);
    if (baselineTaskOptions.length > 0) selectBaselineTaskOption(baselineTaskOptions[0]);
    if (omniretargetTaskOptions.length > 0) selectOmniRetargetTaskOption(omniretargetTaskOptions[0]);
    if (lafanOptions.length > 0) selectLafanOption(lafanOptions[0]);

    function selectPoseOption(selectedOption) {
        // Remove active class from all options
        poseOptions.forEach(option => {
            option.classList.remove('active');
        });
        
        // Add active class to selected option
        selectedOption.classList.add('active');
        
        // Store current pose
        demoState.currentPose = selectedOption.dataset.pose;
        
        // Update description
        const config = poseConfigs[selectedOption.dataset.pose];
        if (config) {
            poseDescription.textContent = config.description;
        }
        
        // Add click effect
        addClickEffect(selectedOption);
        
        // Automatically load the demo when a pose option is selected
        loadPoseDemo();
    }

    function selectSizeOption(selectedOption) {
        // Remove active class from all options
        sizeOptions.forEach(option => {
            option.classList.remove('active');
        });
        
        // Add active class to selected option
        selectedOption.classList.add('active');
        
        // Store current size
        demoState.currentSize = selectedOption.dataset.size;
        
        // Update description
        const config = sizeConfigs[selectedOption.dataset.size];
        if (config) {
            sizeDescription.textContent = config.description;
        }
        
        // Add click effect
        addClickEffect(selectedOption);
        
        // Automatically load the demo when a size option is selected
        loadSizeDemo();
    }

    function updateSizeDescription() {
        const selectedSize = demoState.currentSize;
        const config = sizeConfigs[selectedSize];
        if (config) {
            sizeDescription.textContent = config.description;
        }
    }

    function selectTerrainOption(selectedOption) {
        // Remove active class from all options
        terrainOptions.forEach(option => {
            option.classList.remove('active');
        });
        
        // Add active class to selected option
        selectedOption.classList.add('active');
        
        // Store current terrain
        demoState.currentTerrain = selectedOption.dataset.terrain;
        
        // Update description
        const config = terrainConfigs[selectedOption.dataset.terrain];
        if (config) {
            terrainDescription.textContent = config.description;
        }
        
        // Add click effect
        addClickEffect(selectedOption);
        
        // Automatically load the demo when a terrain option is selected
        loadTerrainDemo();
    }


    function selectEmbodimentOption(selectedOption) {
        // Remove active class from all options
        embodimentOptions.forEach(option => {
            option.classList.remove('active');
        });
        
        // Add active class to selected option
        selectedOption.classList.add('active');
        
        // Store current embodiment
        demoState.currentEmbodiment = selectedOption.dataset.embodiment;
        
        // Update description
        const config = embodimentConfigs[selectedOption.dataset.embodiment];
        if (config) {
            embodimentDescription.textContent = config.description;
        }
        
        // Add click effect
        addClickEffect(selectedOption);
        
        // Automatically load the demo when an embodiment option is selected
        loadEmbodimentDemo();
    }

    function selectBaselineMethodOption(selectedOption) {
        baselineMethodOptions.forEach(option => option.classList.remove('active'));
        selectedOption.classList.add('active');
        demoState.currentBaselineMethod = selectedOption.dataset.baselineMethod;
        addClickEffect(selectedOption);
        loadBaselineMethodDemo();
    }

    function selectBaselineTaskOption(selectedOption) {
        baselineTaskOptions.forEach(option => option.classList.remove('active'));
        selectedOption.classList.add('active');
        demoState.currentBaselineTask = selectedOption.dataset.baselineTask;
        addClickEffect(selectedOption);
        // Keep OmniRetarget task in sync with selected baseline task
        demoState.currentOmniRetargetTask = demoState.currentBaselineTask;
        loadOmniRetargetDemo();
        loadBaselineMethodDemo();
    }

    function selectOmniRetargetTaskOption(selectedOption) {
        omniretargetTaskOptions.forEach(option => option.classList.remove('active'));
        selectedOption.classList.add('active');
        demoState.currentOmniRetargetTask = selectedOption.dataset.omniretargetTask;
        addClickEffect(selectedOption);
        loadOmniRetargetDemo();
    }

    function selectLafanOption(selectedOption) {
        lafanOptions.forEach(option => option.classList.remove('active'));
        selectedOption.classList.add('active');
        demoState.currentLafan = selectedOption.dataset.lafan;
        const config = lafanConfigs[selectedOption.dataset.lafan];
        if (config) lafanDescription.textContent = config.description;
        addClickEffect(selectedOption);
        loadLafanDemo();
    }

    function loadPoseDemo() {
        const selectedPose = demoState.currentPose;
        const config = poseConfigs[selectedPose];
        
        if (config && config.html) {
            const htmlPath = `${filePaths.pose}${config.html}`;
            
            // Add loading indicator
            showLoadingIndicator('pose');
            
            // Force iframe reload by clearing src first
            poseIframe.src = '';
            
            // Small delay to ensure iframe is cleared
            setTimeout(() => {
                poseIframe.src = htmlPath;
            }, 100);

            // Update description
            poseDescription.textContent = config.description;

            // Hide loading indicator after iframe loads
            poseIframe.onload = () => {
                hideLoadingIndicator('pose');
            };

            // Handle loading errors
            poseIframe.onerror = () => {
                hideLoadingIndicator('pose');
                console.error('Failed to load pose demo:', htmlPath);
                poseDescription.textContent = 'Error loading demo. Please try again.';
            };

            // Add loading effect to button
            addLoadingEffect(loadPoseDemoBtn);
        } else {
            console.error('No config found for pose:', selectedPose);
        }
    }

    function loadSizeDemo() {
        const selectedSize = demoState.currentSize;
        const config = sizeConfigs[selectedSize];
        
        if (config && config.html) {
            const htmlPath = `${filePaths.size}${config.html}`;
            
            // Add loading indicator
            showLoadingIndicator('size');
            
            // Force iframe reload by clearing src first
            sizeIframe.src = '';
            
            // Small delay to ensure iframe is cleared
            setTimeout(() => {
                sizeIframe.src = htmlPath;
            }, 100);

            // Update description
            sizeDescription.textContent = config.description;

            // Store current size
            demoState.currentSize = selectedSize;

            // Hide loading indicator after iframe loads
            sizeIframe.onload = () => {
                hideLoadingIndicator('size');
            };

            // Handle loading errors
            sizeIframe.onerror = () => {
                hideLoadingIndicator('size');
                console.error('Failed to load size demo:', htmlPath);
                sizeDescription.textContent = 'Error loading demo. Please try again.';
            };

            // Add loading effect to button
            addLoadingEffect(loadSizeDemoBtn);
        } else {
            console.error('No config found for size:', selectedSize);
        }
    }

    function loadTerrainDemo() {
        const selectedTerrain = demoState.currentTerrain;
        const config = terrainConfigs[selectedTerrain];
        
        if (config && config.html) {
            const htmlPath = `${filePaths.terrain}${config.html}`;
            
            // Add loading indicator
            showLoadingIndicator('terrain');
            
            // Force iframe reload by clearing src first
            terrainIframe.src = '';
            
            // Small delay to ensure iframe is cleared
            setTimeout(() => {
                terrainIframe.src = htmlPath;
            }, 100);

            // Update description
            terrainDescription.textContent = config.description;

            // Hide loading indicator after iframe loads
            terrainIframe.onload = () => {
                hideLoadingIndicator('terrain');
            };

            // Handle loading errors
            terrainIframe.onerror = () => {
                hideLoadingIndicator('terrain');
                console.error('Failed to load terrain demo:', htmlPath);
                terrainDescription.textContent = 'Error loading demo. Please try again.';
            };

            // Add loading effect to button
            addLoadingEffect(loadTerrainDemoBtn);
        } else {
            console.error('No config found for terrain:', selectedTerrain);
        }
    }

    function loadEmbodimentDemo() {
        const selectedEmbodiment = demoState.currentEmbodiment;
        const config = embodimentConfigs[selectedEmbodiment];
        
        if (config && config.html) {
            const htmlPath = `${filePaths.embodiment}${config.html}`;
            
            // Add loading indicator
            showLoadingIndicator('embodiment');
            
            // Force iframe reload by clearing src first
            embodimentIframe.src = '';
            
            // Small delay to ensure iframe is cleared
            setTimeout(() => {
                embodimentIframe.src = htmlPath;
            }, 100);

            // Update description
            embodimentDescription.textContent = config.description;

            // Hide loading indicator after iframe loads
            embodimentIframe.onload = () => {
                hideLoadingIndicator('embodiment');
            };

            // Handle loading errors
            embodimentIframe.onerror = () => {
                hideLoadingIndicator('embodiment');
                console.error('Failed to load embodiment demo:', htmlPath);
                embodimentDescription.textContent = 'Error loading demo. Please try again.';
            };

            // Add loading effect to button
            addLoadingEffect(loadEmbodimentDemoBtn);
        } else {
            console.error('No config found for embodiment:', selectedEmbodiment);
        }
    }

    function loadBaselineMethodDemo() {
        const method = demoState.currentBaselineMethod || 'gmr';
        const task = demoState.currentBaselineTask || 'box';
        const selected = `${method}_${task}`;
        const config = baselineConfigs[selected];
        if (config && config.html) {
            const htmlPath = `${filePaths.baseline}${config.html}`;
            showLoadingIndicator('baselineMethod');
            baselineMethodIframe.src = '';
            setTimeout(() => {
                baselineMethodIframe.src = htmlPath;
            }, 100);
            baselineMethodDescription.textContent = config.description;
            updateBaselineTitles();
            baselineMethodIframe.onload = () => hideLoadingIndicator('baselineMethod');
            baselineMethodIframe.onerror = () => {
                hideLoadingIndicator('baselineMethod');
                console.error('Failed to load baseline method demo:', htmlPath);
                baselineMethodDescription.textContent = 'Error loading demo. Please try again.';
            };
            addLoadingEffect(loadBaselineMethodDemoBtn);
        } else {
            console.error('No config found for baseline method:', selected);
        }
    }

    function loadOmniRetargetDemo() {
        const task = demoState.currentOmniRetargetTask || 'box';
        const selected = `omniretarget_${task}`;
        const config = baselineConfigs[selected];
        if (config && config.html) {
            const htmlPath = `${filePaths.baseline}${config.html}`;
            showLoadingIndicator('omniretarget');
            omniretargetIframe.src = '';
            setTimeout(() => {
                omniretargetIframe.src = htmlPath;
            }, 100);
            omniretargetDescription.textContent = config.description;
            updateBaselineTitles();
            omniretargetIframe.onload = () => hideLoadingIndicator('omniretarget');
            omniretargetIframe.onerror = () => {
                hideLoadingIndicator('omniretarget');
                console.error('Failed to load omniretarget demo:', htmlPath);
                omniretargetDescription.textContent = 'Error loading demo. Please try again.';
            };
            if (typeof loadOmniRetargetDemoBtn !== 'undefined' && loadOmniRetargetDemoBtn) {
                addLoadingEffect(loadOmniRetargetDemoBtn);
            }
        } else {
            console.error('No config found for omniretarget:', selected);
        }
    }

    function updateBaselineTitles() {
        const method = (demoState.currentBaselineMethod || 'gmr').toUpperCase();
        const task = demoState.currentBaselineTask || 'box';
        const taskLabel = task === 'box' ? 'Object Carrying' : 'Climbing';
        if (baselineMethodTitle) {
            baselineMethodTitle.innerHTML = `<i class="fas fa-brain" style="margin-right: 0.5rem;"></i> ${method} - ${taskLabel}`;
        }
        if (omniretargetTitle) {
            omniretargetTitle.innerHTML = `<i class="fas fa-check" style="margin-right: 0.5rem;"></i> OmniRetarget - ${taskLabel}`;
        }
    }

    function loadLafanDemo() {
        const selected = demoState.currentLafan || 'dance1_subject1';
        const config = lafanConfigs[selected];
        if (config && config.html) {
            const htmlPath = `${filePaths.lafan}${config.html}`;
            showLoadingIndicator('lafan');
            lafanIframe.src = '';
            setTimeout(() => {
                lafanIframe.src = htmlPath;
            }, 100);
            lafanDescription.textContent = config.description;
            lafanIframe.onload = () => hideLoadingIndicator('lafan');
            lafanIframe.onerror = () => {
                hideLoadingIndicator('lafan');
                console.error('Failed to load lafan demo:', htmlPath);
                lafanDescription.textContent = 'Error loading demo. Please try again.';
            };
            addLoadingEffect(loadLafanDemoBtn);
        } else {
            console.error('No config found for lafan:', selected);
        }
    }

    function addClickEffect(element) {
        element.style.transform = 'scale(0.95)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 150);
    }

    function toggleFullscreen(demoType = 'both') {
        const targetIframe = demoType === 'pose' ? poseIframe : 
                           demoType === 'size' ? sizeIframe :
                           demoType === 'terrain' ? terrainIframe :
                           demoType === 'embodiment' ? embodimentIframe :
                           demoType === 'baselineMethod' ? baselineMethodIframe :
                           demoType === 'omniretarget' ? omniretargetIframe :
                           demoType === 'lafan' ? lafanIframe : null;
        
        if (!demoState.isFullscreen) {
            // Enter fullscreen - target only the iframe
            if (targetIframe) {
                // Create a fullscreen container for the iframe
                const fullscreenContainer = document.createElement('div');
                fullscreenContainer.id = 'fullscreen-iframe-container';
                fullscreenContainer.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background: #000;
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                `;
                
                // Clone the iframe for fullscreen
                const fullscreenIframe = targetIframe.cloneNode(true);
                fullscreenIframe.style.cssText = `
                    width: 100vw;
                    height: 100vh;
                    border: none;
                    border-radius: 0;
                `;
                
                // Add close button
                const closeButton = document.createElement('button');
                closeButton.innerHTML = '✕';
                closeButton.style.cssText = `
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    border: none;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    font-size: 20px;
                    cursor: pointer;
                    z-index: 10000;
                `;
                
                closeButton.addEventListener('click', () => {
                    exitFullscreen();
                });
                
                // Add keyboard support for escape key
                const handleKeyPress = (e) => {
                    if (e.key === 'Escape') {
                        exitFullscreen();
                        document.removeEventListener('keydown', handleKeyPress);
                    }
                };
                document.addEventListener('keydown', handleKeyPress);
                
                fullscreenContainer.appendChild(fullscreenIframe);
                fullscreenContainer.appendChild(closeButton);
                document.body.appendChild(fullscreenContainer);
                
                // Request fullscreen
                if (fullscreenContainer.requestFullscreen) {
                    fullscreenContainer.requestFullscreen();
                } else if (fullscreenContainer.webkitRequestFullscreen) {
                    fullscreenContainer.webkitRequestFullscreen();
                } else if (fullscreenContainer.msRequestFullscreen) {
                    fullscreenContainer.msRequestFullscreen();
                }
                
                demoState.isFullscreen = true;
                demoState.fullscreenType = demoType;
                demoState.fullscreenContainer = fullscreenContainer;
                
                // Update button states
                updateFullscreenButtons(true, demoType);
            }
        } else {
            exitFullscreen();
        }
    }

    function exitFullscreen() {
        // Exit fullscreen
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        
        // Remove fullscreen container
        if (demoState.fullscreenContainer) {
            document.body.removeChild(demoState.fullscreenContainer);
            demoState.fullscreenContainer = null;
        }
        
        demoState.isFullscreen = false;
        demoState.fullscreenType = null;
        
        // Update button states
        updateFullscreenButtons(false, 'both');
    }

    function updateFullscreenButtons(isFullscreen, demoType) {
        const expandIcon = 'fas fa-expand';
        const compressIcon = 'fas fa-compress';
        
        if (isFullscreen) {
            if (demoType === 'pose' || demoType === 'both') {
                if (fullscreenPoseBtn) {
                    fullscreenPoseBtn.querySelector('i').className = compressIcon;
                    fullscreenPoseBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'size' || demoType === 'both') {
                if (fullscreenSizeBtn) {
                    fullscreenSizeBtn.querySelector('i').className = compressIcon;
                    fullscreenSizeBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'terrain' || demoType === 'both') {
                if (fullscreenTerrainBtn) {
                    fullscreenTerrainBtn.querySelector('i').className = compressIcon;
                    fullscreenTerrainBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'embodiment' || demoType === 'both') {
                if (fullscreenEmbodimentBtn) {
                    fullscreenEmbodimentBtn.querySelector('i').className = compressIcon;
                    fullscreenEmbodimentBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'baselineMethod' || demoType === 'both') {
                if (fullscreenBaselineMethodBtn) {
                    fullscreenBaselineMethodBtn.querySelector('i').className = compressIcon;
                    fullscreenBaselineMethodBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'omniretarget' || demoType === 'both') {
                if (fullscreenOmniRetargetBtn) {
                    fullscreenOmniRetargetBtn.querySelector('i').className = compressIcon;
                    fullscreenOmniRetargetBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (demoType === 'lafan' || demoType === 'both') {
                if (fullscreenLafanBtn) {
                    fullscreenLafanBtn.querySelector('i').className = compressIcon;
                    fullscreenLafanBtn.querySelector('span:last-child').textContent = 'Exit';
                }
            }
            if (fullscreenBtn) {
                fullscreenBtn.querySelector('i').className = compressIcon;
                fullscreenBtn.querySelector('span:last-child').textContent = 'Exit Fullscreen';
            }
        } else {
            if (fullscreenPoseBtn) {
                fullscreenPoseBtn.querySelector('i').className = expandIcon;
                fullscreenPoseBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenSizeBtn) {
                fullscreenSizeBtn.querySelector('i').className = expandIcon;
                fullscreenSizeBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenTerrainBtn) {
                fullscreenTerrainBtn.querySelector('i').className = expandIcon;
                fullscreenTerrainBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenEmbodimentBtn) {
                fullscreenEmbodimentBtn.querySelector('i').className = expandIcon;
                fullscreenEmbodimentBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenBaselineBoxBtn) {
                fullscreenBaselineBoxBtn.querySelector('i').className = expandIcon;
                fullscreenBaselineBoxBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenBaselineClimbBtn) {
                fullscreenBaselineClimbBtn.querySelector('i').className = expandIcon;
                fullscreenBaselineClimbBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenLafanBtn) {
                fullscreenLafanBtn.querySelector('i').className = expandIcon;
                fullscreenLafanBtn.querySelector('span:last-child').textContent = 'Fullscreen';
            }
            if (fullscreenBtn) {
                fullscreenBtn.querySelector('i').className = expandIcon;
                fullscreenBtn.querySelector('span:last-child').textContent = 'Fullscreen View';
            }
        }
    }

    function showLoadingIndicator(type) {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = `demo-loading-${type}`;
        loadingDiv.innerHTML = `
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 10px; 
                        text-align: center; z-index: 1000;">
                <i class="fas fa-spinner fa-spin" style="font-size: 1.5rem; color: #3498db; margin-bottom: 0.5rem;"></i>
                <p style="color: #333; font-weight: bold; font-size: 0.9rem;">Loading ${type} demo...</p>
            </div>
        `;
        loadingDiv.style.position = 'relative';
        
        const targetWrapper = type === 'pose' ? poseIframe.parentElement : 
            type === 'size' ? sizeIframe.parentElement :
            type === 'terrain' ? terrainIframe.parentElement :
            type === 'embodiment' ? embodimentIframe.parentElement :
            type === 'baselineMethod' ? baselineMethodIframe.parentElement :
            type === 'omniretarget' ? omniretargetIframe.parentElement :
            lafanIframe.parentElement;
        targetWrapper.appendChild(loadingDiv);
    }

    function hideLoadingIndicator(type) {
        const loadingDiv = document.getElementById(`demo-loading-${type}`);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    function addLoadingEffect(element) {
        element.style.opacity = '0.7';
        element.style.transform = 'scale(0.95)';
        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'scale(1)';
        }, 300);
    }

    // Add interactive effects
    function addInteractiveEffects() {
        // Add hover effects to size dropdown
        sizeSelect.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
            this.parentElement.style.transition = 'transform 0.3s ease';
        });
        
        sizeSelect.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });

        // Add iframe hover effects
        [poseIframe, sizeIframe].forEach(iframe => {
            iframe.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.01)';
                this.style.transition = 'transform 0.3s ease';
            });

            iframe.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });

        // Handle fullscreen change events
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('msfullscreenchange', handleFullscreenChange);
    }

    function handleFullscreenChange() {
        if (!document.fullscreenElement && !document.webkitFullscreenElement && !document.msFullscreenElement) {
            // Exited fullscreen
            exitFullscreen();
        }
    }

    // Initialize interactive effects
    addInteractiveEffects();
}

// Baseline and LAFAN specific logic
function initializeBaselineAndLafan() {
    // This function is intentionally left in case we need future separation
}
