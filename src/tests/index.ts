// Ensure globals are defined before importing specs
import '../test-utils/browserTestRunner.js';

// Import all spec files
import '../browser.spec.js';
import './shader-extensions/test.spec.js';
import '../gpu/256bit/pallas_msm.spec.js'