import { expect } from 'chai';
import staticFrag from './test.wgsl';
import composedFrag from './test.wgslc.js';
import dynamicFrag from './test.wgsld.js';

describe('Shader Extensions', () => {

    it('imports static WGSL exactly', () => {
        const expected = `// static WGSL fragment
fn staticFunc() -> void { }`;
        expect(staticFrag.trim()).to.equal(expected);
    });

    it('imports composed WGSL exactly', () => {
        const expected = `// static WGSL fragment
fn staticFunc() -> void { }

fn composedFunc() -> void { }`;
        expect(composedFrag.trim()).to.equal(expected);
    });

    it('imports dynamic WGSL function exactly', () => {
        const expectedTrue = `// static WGSL fragment
fn staticFunc() -> void { }

fn dynamicFunc() -> void {
    // extra code here
}`;
        const expectedFalse = `// static WGSL fragment
fn staticFunc() -> void { }

fn dynamicFunc() -> void {
    
}`;

        expect(dynamicFrag(true).trim()).to.equal(expectedTrue);
        expect(dynamicFrag(false).trim()).to.equal(expectedFalse);
    });

});
