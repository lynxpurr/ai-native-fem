"use strict";

const get = id => document.getElementById(id);
const scene = JSON.parse(get("result-data").textContent);
const fieldSelect = get("field");
const scaleInput = get("scale");
const canvas = get("mesh");
const modal = scene.kind === "modal_result";
const solved = modal || scene.solve_status === "solved";
const modeSelect = get("mode");
const play = get("play");
const plate = scene.problem === "rectangular_plate";

get("model").textContent = scene.problem.replaceAll("_", " ") + " · schema " + scene.schema_version;
get("status").textContent = modal ? "Modal result · " + scene.rigid_mode_count + " rigid modes"
    + (scene.truncated ? " · truncated" : "") : scene.solve_status + " · " + scene.verification_status;
get("status").dataset.warning = modal ? scene.truncated : scene.verification_status !== "verification_passed";
get("geometry").textContent = scene.coordinates + " · geometry in m";
for (const diagnostic of scene.diagnostics) {
    const item = document.createElement("li");
    item.textContent = modal ? diagnostic : diagnostic.code + ": " + diagnostic.message;
    get("diagnostics").append(item);
}
if (modal) {
    get("mode-control").hidden = false;
    get("mode-info").hidden = false;
    play.hidden = false;
    get("scale-label").textContent = "Relative amplitude (×)";
    get("deformed-key").textContent = "Mode shape";
    scene.frequencies_hz.forEach((frequency, index) => {
        modeSelect.add(new Option("Mode " + (index + 1) + " (" + frequency.toPrecision(6) + " Hz)",
            String(index)));
    });
}
if (!solved) {
    fieldSelect.add(new Option("Unavailable", ""));
    fieldSelect.disabled = true;
    scaleInput.value = "0";
    scaleInput.disabled = true;
    get("deformed-key").hidden = true;
    get("field-values").hidden = true;
    get("field-info").textContent = "Result fields and deformation are unavailable.";
}

function setRows(id, rows) {
    const elements = rows.map(values => {
        const row = document.createElement("tr");
        for (const value of values) {
            const cell = document.createElement("td");
            cell.textContent = String(value);
            row.append(cell);
        }
        return row;
    });
    get(id).replaceChildren(...elements);
}

const loads = scene.loads.filter(load => load[4] !== 0);
setRows("loads", loads.length ? loads.map(load => load.slice(1)) : [["No nonzero loads"]]);
setRows("supports", scene.supports.length
    ? scene.supports.map(([index, ...values]) => [scene.labels[index], ...values])
    : [["No prescribed supports"]]);
for (const [label, value] of [
    ["Input SHA-256", scene.input_hash],
    ...(modal ? [] : [
        ["Strain energy (J)", scene.strain_energy],
        ["Relative residual", scene.relative_residual],
    ]),
]) {
    const term = document.createElement("dt");
    const definition = document.createElement("dd");
    term.textContent = label;
    definition.textContent = value ?? "Unavailable";
    get("details").append(term, definition);
}

function samples(field) {
    if (field.position === "member_end") {
        return scene.member_ends.flatMap(([name, first, last], index) => [
            {label: name + " / i (" + scene.labels[first[0]] + ")", location: [...first, 0.15],
                value: field.values[2 * index]},
            {label: name + " / j (" + scene.labels[last[0]] + ")", location: [...last, 0.15],
                value: field.values[2 * index + 1]},
        ]);
    }
    const nodes = field.nodes ?? (field.position === "frame_node" ? scene.frame_nodes
        : scene.labels.map((_, i) => i));
    return field.values.map((value, index) => {
        const node = nodes[index];
        return {label: scene.labels[node], location: node, value};
    });
}

function movePoint(point, displacement, scale) {
    return point.map((value, axis) => value + scale * displacement[axis]);
}

function interpolate(points, location) {
    if (!Array.isArray(location)) return points[location];
    const [i, j, fraction = 0.5] = location;
    return points[i].map((value, axis) => value + fraction * (points[j][axis] - value));
}

function modalFrame(points, motion, amplitude, phase) {
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const scale = 0.1 * span * amplitude * Math.cos(phase);
    return points.map((point, index) => movePoint(point, motion[index], scale));
}

let fields = scene.fields;
let modeMotion = [];
let phase = 0;
let animationStart = null;
let animationFrame = null;

function selectMode() {
    const selected = fieldSelect.value;
    const index = modeSelect.selectedIndex;
    if (modal) {
        const mode = scene.modes[index];
        fields = mode.fields;
        modeMotion = mode.motion;
        phase = 0;
        animationStart = null;
        get("mode-info").textContent = scene.frequencies_hz[index] + " Hz · relative residual "
            + scene.residuals[index] + " · amplitude 1 is 10% of the mesh span"
            + " · illustrative playback at a fixed period";
    }
    fieldSelect.replaceChildren(...fields.map(field => new Option(field.name, field.name)));
    if (fields.some(field => field.name === selected)) fieldSelect.value = selected;
    selectField();
}

function animate(now) {
    if (animationStart === null) animationStart = now - phase * 2000 / (2 * Math.PI);
    phase = ((now - animationStart) % 2000) * 2 * Math.PI / 2000;
    draw();
    animationFrame = requestAnimationFrame(animate);
}

let displayedSamples = [];
let minimum = 0;
let maximum = 0;
function selectField() {
    if (!solved) return;
    const field = fields[fieldSelect.selectedIndex];
    displayedSamples = samples(field);
    minimum = Math.min(...field.values);
    maximum = Math.max(...field.values);
    get("field-info").textContent = field.name + " · " + field.unit + " · "
        + field.position + " · " + field.coordinates;
    get("range").textContent = "Sample range " + minimum + " to " + maximum + " " + field.unit;
    get("field-unit").textContent = "Value (" + field.unit + ")";
    setRows("values", displayedSamples.map(sample => [sample.label, sample.value]));
    draw();
}

function draw() {
    const {width, height} = canvas.getBoundingClientRect();
    if (width === 0 || height === 0) return;
    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const scale = scaleInput.validity.valid && scaleInput.value !== "" ? scaleInput.valueAsNumber : 0;
    const displacement = scene.displacement ?? [];
    // Normalize before projecting so unusually large finite values remain drawable.
    const extent = Math.max(Number.MIN_VALUE, ...scene.points.flat().map(Math.abs),
        ...(scale === 0 ? [] : displacement.flat().map(Math.abs)));
    const origin = scene.points[0].map(value => value / extent);
    const normalized = scene.points.map(point => point.map((value, axis) => value / extent - origin[axis]));
    const project = ([x, y, z]) => plate ? [x - 0.52 * y, 0.28 * x + 0.56 * y + z] : [x, y];
    const original = normalized.map(project);
    let deformed = original;
    if (scale > 0) {
        deformed = modal ? modalFrame(normalized, modeMotion, scale, phase).map(project)
            : normalized.map((point, i) =>
                project(movePoint(point, displacement[i].map(value => value / extent), scale)));
    }
    // Bound modal playback at both peaks so its viewport stays fixed through a cycle.
    const all = modal && scale > 0 ? [
        ...original,
        ...modalFrame(normalized, modeMotion, scale, 0).map(project),
        ...modalFrame(normalized, modeMotion, scale, Math.PI).map(project),
    ] : [...original, ...deformed];
    const xs = all.map(point => point[0]);
    const ys = all.map(point => point[1]);
    const left = Math.min(...xs), right = Math.max(...xs);
    const bottom = Math.min(...ys), top = Math.max(...ys);
    const span = Math.max(right - left, top - bottom) || 1;
    const pixels = Math.min(
        (width - 48) / ((right - left) / span || 1e-9),
        (height - 48) / ((top - bottom) / span || 1e-9),
    );
    const screen = ([x, y]) => [width / 2 + pixels * ((x - (left + right) / 2) / span),
        height / 2 - pixels * ((y - (bottom + top) / 2) / span)];

    function mesh(points, color, lineWidth) {
        context.beginPath();
        for (const cell of scene.cells) {
            context.moveTo(...screen(points[cell[0]]));
            for (const node of cell.slice(1)) context.lineTo(...screen(points[node]));
            if (cell.length > 2) context.closePath();
        }
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        context.stroke();
    }
    mesh(original, "#8494a8", 2);
    if (scale > 0) mesh(deformed, "#173e69", 1.5);

    context.fillStyle = "#0a715d";
    for (const index of new Set(scene.supports.map(row => row[0]))) {
        const [x, y] = screen(original[index]);
        context.fillRect(x - 4, y - 4, 8, 8);
    }
    const loadLocations = new Map(loads.map(([location]) => [JSON.stringify(location), location]));
    for (const location of loadLocations.values()) {
        const point = interpolate(original, location ?? [0, original.length - 1]);
        const [x, y] = screen(point);
        context.strokeStyle = "#b84613";
        context.lineWidth = 2;
        context.strokeRect(x - 6, y - 6, 12, 12);
    }
    const bound = Math.max(Math.abs(minimum), Math.abs(maximum)) || 1;
    const low = minimum / bound, high = maximum / bound;
    for (const sample of displayedSamples) {
        const fraction = high === low ? 0.5 : (sample.value / bound - low) / (high - low);
        const [x, y] = screen(interpolate(deformed, sample.location));
        context.beginPath();
        context.arc(x, y, 3, 0, 2 * Math.PI);
        context.fillStyle = "hsl(" + (220 - 205 * fraction) + " 78% 43%)";
        context.fill();
    }
}

fieldSelect.addEventListener("change", selectField);
scaleInput.addEventListener("input", draw);
scaleInput.addEventListener("change", () => {
    if (!scaleInput.validity.valid || scaleInput.value === "") {
        scaleInput.value = "0";
        draw();
    }
});
modeSelect.addEventListener("change", selectMode);
play.addEventListener("click", () => {
    if (animationFrame === null) {
        play.textContent = "Pause animation";
        animationFrame = requestAnimationFrame(animate);
    } else {
        cancelAnimationFrame(animationFrame);
        animationFrame = animationStart = null;
        play.textContent = "Play animation";
    }
});
new ResizeObserver(draw).observe(canvas);
if (solved) selectMode();
