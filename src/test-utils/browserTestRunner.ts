interface Test {
    name: string;
    fn: () => void | Promise<void>;
}

const tests: Test[] = [];
let afterAllFn: (() => void) | undefined;

export function describe(name: string, fn: () => void) {
    fn();
}

export function it(name: string, fn: () => void | Promise<void>) {
    tests.push({ name, fn });
}

export function after(fn: () => void) {
    afterAllFn = fn;
}

export async function runTests() {
    const container = document.createElement('div');
    container.id = 'test-results';
    document.body.appendChild(container);

    let failures = 0;

    for (const t of tests) {
        try {
            await t.fn();
            console.log(`✅ ${t.name}`);
            container.innerHTML += `<div style="color:green;">✅ ${t.name}</div>`;
        } catch (err) {
            failures++;
            console.error(`❌ ${t.name}`, err);
            container.innerHTML += `<div style="color:red;">❌ ${t.name}: ${err}</div>`;
        }
    }

    if (afterAllFn) afterAllFn();

    // Signal Puppeteer that tests finished
    (globalThis as any).testsFinished = true;
    (globalThis as any).testsFailures = failures;
}

// Attach to globalThis **before any specs are imported**
(globalThis as any).describe = describe;
(globalThis as any).it = it;
(globalThis as any).after = after;
(globalThis as any).runTests = runTests;
