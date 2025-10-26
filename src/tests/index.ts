// Ensure globals are defined before importing specs
import '../test-utils/browserTestRunner.js';

// Import all spec files
import '../browser.spec.js';
import './shader-extensions/test.spec.js';
import '../gpu/256bit/pallas/mul_add_carry.spec.js';
import '../gpu/256bit/pallas/to_montgomery.spec.js';
import '../gpu/256bit/pallas/constants.spec.js';
import '../gpu/256bit/pallas/add.spec.js';
import '../gpu/256bit/pallas/sub.spec.js';
import '../gpu/256bit/pallas/mul.spec.js';
import '../gpu/256bit/pallas/msm.spec.js';
import '../gpu/256bit/pallas/pippenger_msm.spec.js';
