import frag from './test.wgsl';

export default function dynamicShader(extra: boolean) {
    return `
${frag}

fn dynamicFunc() -> void {
    ${extra ? '// extra code here' : ''}
}
`;
}