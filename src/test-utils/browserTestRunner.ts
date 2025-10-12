interface Test {
    name: string;
    fn: () => void | Promise<void>;
}

const tests: Test[] = [];

// Hooks
let beforeAllFn: (() => void | Promise<void>) | undefined;
let beforeEachFn: (() => void | Promise<void>) | undefined;
let afterEachFn: (() => void | Promise<void>) | undefined;
let afterAllFn: (() => void | Promise<void>) | undefined;

export function describe(name: string, fn: () => void) {
    fn();
}

export function it(name: string, fn: () => void | Promise<void>) {
    tests.push({ name, fn });
}

// Hooks API
export function beforeAll(fn: () => void | Promise<void>) { beforeAllFn = fn; }
export function beforeEach(fn: () => void | Promise<void>) { beforeEachFn = fn; }
export function afterEach(fn: () => void | Promise<void>) { afterEachFn = fn; }
export function after(fn: () => void | Promise<void>) { afterAllFn = fn; }

export async function runTests() {
    const container = document.createElement('div');
    container.id = 'test-results';
    document.body.appendChild(container);

    let failures = 0;

    if (beforeAllFn) await beforeAllFn();

    for (const t of tests) {
        if (beforeEachFn) await beforeEachFn();

        try {
            await t.fn();
            console.log(`✅ ${t.name}`);
            container.innerHTML += `<div style="color:green;">✅ ${t.name}</div>`;
        } catch (err) {
            failures++;
            console.error(`❌ ${t.name}`, err);
            container.innerHTML += `<div style="color:red;">❌ ${t.name}: ${err}</div>`;
        }

        if (afterEachFn) await afterEachFn();
    }

    if (afterAllFn) await afterAllFn();

    // Signal Puppeteer that tests finished
    (globalThis as any).testsFinished = true;
    (globalThis as any).testsFailures = failures;
}

// Attach to globalThis **before any specs are imported**
(globalThis as any).describe = describe;
(globalThis as any).it = it;
(globalThis as any).beforeAll = beforeAll;
(globalThis as any).beforeEach = beforeEach;
(globalThis as any).afterEach = afterEach;
(globalThis as any).after = after;
(globalThis as any).runTests = runTests;