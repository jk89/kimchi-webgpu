interface Test {
  name: string;
  fn: (() => void | Promise<void>) | null;
  timeout?: number;
  mode?: "normal" | "skip" | "only";
}

interface Suite {
  name: string;
  tests: Test[];
  beforeAllFns: (() => void | Promise<void>)[];
  beforeEachFns: (() => void | Promise<void>)[];
  afterEachFns: (() => void | Promise<void>)[];
  afterAllFns: (() => void | Promise<void>)[];
  suites: Suite[];
}

let currentSuite: Suite = {
  name: "root",
  tests: [],
  beforeAllFns: [],
  beforeEachFns: [],
  afterEachFns: [],
  afterAllFns: [],
  suites: [],
};

const rootSuite = currentSuite;
let hasOnly = false;

// === Definition API ===

export function describe(name: string, fn: () => void) {
  const parent = currentSuite;
  const suite: Suite = {
    name,
    tests: [],
    beforeAllFns: [],
    beforeEachFns: [],
    afterEachFns: [],
    afterAllFns: [],
    suites: [],
  };

  parent.suites.push(suite);
  currentSuite = suite;
  fn();
  currentSuite = parent;
}

function addTest(
  name: string,
  fn: (() => void | Promise<void>) | null,
  timeout = Infinity,
  mode: "normal" | "skip" | "only" = "normal"
) {
  if (mode === "only") hasOnly = true;
  currentSuite.tests.push({ name, fn, timeout, mode });
}

export function it(name: string, fn: () => void | Promise<void>, timeout = Infinity) {
  addTest(name, fn, timeout, "normal");
}
export const test = it;

it.skip = (name: string, _fn?: () => void | Promise<void>) => addTest(name, null, 0, "skip");
it.only = (name: string, fn: () => void | Promise<void>, timeout = Infinity) =>
  addTest(name, fn, timeout, "only");
test.skip = it.skip;
test.only = it.only;

// === Hooks ===
export function beforeAll(fn: () => void | Promise<void>) {
  currentSuite.beforeAllFns.push(fn);
}
export function beforeEach(fn: () => void | Promise<void>) {
  currentSuite.beforeEachFns.push(fn);
}
export function afterEach(fn: () => void | Promise<void>) {
  currentSuite.afterEachFns.push(fn);
}
export function after(fn: () => void | Promise<void>) {
  currentSuite.afterAllFns.push(fn);
}

// === Helpers ===
function withTimeout<T>(promise: Promise<T>, ms: number, testName: string): Promise<T> {
  if (ms === Infinity) return promise; // skip timeout
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Test timed out after ${ms} ms: ${testName}`)), ms);
    promise
      .then((v) => {
        clearTimeout(timer);
        resolve(v);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

// === Runner ===
async function runSuite(
  suite: Suite,
  depth = 0,
  parentHooks?: {
    beforeEach: (() => void | Promise<void>)[];
    afterEach: (() => void | Promise<void>)[];
  },
  parentEl?: HTMLElement
) {
  const indent = "  ".repeat(depth);
  const beforeEachChain = [...(parentHooks?.beforeEach ?? []), ...suite.beforeEachFns];
  const afterEachChain = [...suite.afterEachFns, ...(parentHooks?.afterEach ?? [])];

  const suiteEl = document.createElement("section");
  suiteEl.style.marginLeft = parentEl ? "20px" : "0";
  suiteEl.style.padding = "8px";
  suiteEl.style.borderLeft = parentEl ? "2px solid #ccc" : "none";

  const title = document.createElement("h3");
  title.textContent = suite.name;
  title.style.fontFamily = "monospace";
  title.style.color = "#333";
  suiteEl.appendChild(title);
  (parentEl ?? document.body).appendChild(suiteEl);

  console.log(`${indent}${suite.name}`);

  for (const fn of suite.beforeAllFns) await fn();

  const runnableTests = hasOnly
    ? suite.tests.filter((t) => t.mode === "only")
    : suite.tests;

  for (const t of runnableTests) {
    const testEl = document.createElement("div");
    testEl.style.marginLeft = "16px";
    testEl.style.fontFamily = "monospace";
    suiteEl.appendChild(testEl);

    const prefix = indent + "  ";

    if (t.mode === "skip" || t.fn === null) {
      console.log(`${prefix}⏭️  ${t.name}`);
      testEl.textContent = `⏭️  ${t.name} (skipped)`;
      testEl.style.color = "gray";
      continue;
    }

    const testStart = performance.now();
    try {
      for (const fn of beforeEachChain) await fn();

      await withTimeout(Promise.resolve(t.fn()), t.timeout!, t.name);

      const testEnd = performance.now();
      const duration = (testEnd - testStart).toFixed(2);

      console.log(`${prefix}✅ ${t.name} (${duration} ms)`);
      testEl.textContent = `✅ ${t.name} (${duration} ms)`;
      testEl.style.color = "green";
    } catch (err) {
      const testEnd = performance.now();
      const duration = (testEnd - testStart).toFixed(2);

      console.error(`${prefix}❌ ${t.name} (${duration} ms)`, err);
      testEl.textContent = `❌ ${t.name} (${duration} ms): ${err}`;
      testEl.style.color = "red";
    } finally {
      for (const fn of afterEachChain) await fn();
    }
  }

  for (const child of suite.suites) {
    await runSuite(child, depth + 1, { beforeEach: beforeEachChain, afterEach: afterEachChain }, suiteEl);
  }

  for (const fn of suite.afterAllFns) await fn();
}

export async function runTests() {
  const container = document.createElement("div");
  container.id = "test-results";
  container.style.fontFamily = "sans-serif";
  container.style.fontSize = "14px";
  container.style.lineHeight = "1.4";
  container.style.margin = "20px";
  document.body.appendChild(container);

  const startTime = performance.now();
  await runSuite(rootSuite, 0, undefined, container);
  const endTime = performance.now();

  const totalDuration = (endTime - startTime).toFixed(2);
  console.log(`Total test run time: ${totalDuration} ms`);

  const timeEl = document.createElement("div");
  timeEl.style.marginTop = "20px";
  timeEl.style.fontWeight = "bold";
  timeEl.textContent = `Total test run time: ${totalDuration} ms`;
  container.appendChild(timeEl);

  (globalThis as any).testsFinished = true;
}

// === Globals ===
(globalThis as any).describe = describe;
(globalThis as any).it = it;
(globalThis as any).test = test;
(globalThis as any).beforeAll = beforeAll;
(globalThis as any).beforeEach = beforeEach;
(globalThis as any).afterEach = afterEach;
(globalThis as any).after = after;
(globalThis as any).runTests = runTests;