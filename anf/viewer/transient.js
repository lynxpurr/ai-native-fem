"use strict";

const get = id => document.getElementById(id);
const scene = JSON.parse(get("result-data").textContent);
const nodeSelect = get("node");
const componentSelect = get("component");
const sample = get("sample");
const canvas = get("history");
let series;
let peakIndex = 0;

get("time-grid").textContent = "Schema 1.5 · " + scene.point_count + " samples · step " + scene.dt_s + " s";
get("first-time").textContent = scene.times[0];
get("last-time").textContent = scene.times.at(-1);
sample.max = String(scene.point_count - 1);

for (const [index, node] of scene.nodes.entries()) {
    nodeSelect.add(new Option(node.label, String(index)));
}
for (const diagnostic of scene.diagnostics) {
    const row = document.createElement("li");
    row.textContent = diagnostic;
    get("diagnostics").append(row);
}
for (const [name, value] of [
    ["Input SHA-256", scene.input_hash],
    ["Worst relative backward error", scene.worst_backward_error],
    ["Step with worst backward error", scene.worst_backward_step],
]) {
    const key = document.createElement("dt");
    const item = document.createElement("dd");
    key.textContent = name;
    item.textContent = value;
    get("metadata").append(key, item);
}

function selectNode() {
    const previous = componentSelect.value;
    const components = scene.nodes[nodeSelect.selectedIndex].components;
    componentSelect.replaceChildren(...components.map(dof => new Option(dof.name, dof.name)));
    if (components.some(dof => dof.name === previous)) componentSelect.value = previous;
    selectComponent();
}

function selectComponent() {
    series = scene.nodes[nodeSelect.selectedIndex].components[componentSelect.selectedIndex];
    peakIndex = scene.times.indexOf(series.peak_time);
    const axisName = series.unit === "rad" ? "Rotation" : "Displacement";
    get("axis-y").textContent = axisName + " (" + series.unit + ")";
    canvas.setAttribute("aria-label", axisName + " versus time. Use Sample index to read an exact value.");
    get("peak").textContent = "Absolute peak " + series.peak + " " + series.unit
        + " at " + series.peak_time + " s · signed value " + series.values[peakIndex] + " " + series.unit;
    showSample();
    draw();
}

function showSample() {
    if (!sample.validity.valid || sample.value === "") {
        get("sample-value").textContent = "Choose a sample from 0 to " + sample.max + ".";
        return;
    }
    const index = sample.valueAsNumber;
    get("sample-value").textContent = scene.times[index] + " s · " + series.values[index] + " " + series.unit;
}

function draw() {
    const {width, height} = canvas.getBoundingClientRect();
    if (width <= 85 || height < 60) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const start = scene.times[0], span = scene.times.at(-1) - start;
    const low = Math.min(...series.values), high = Math.max(...series.values);
    const magnitude = Math.max(Math.abs(low), Math.abs(high)) || 1;
    const bottom = low / magnitude, top = high / magnitude;
    const leftMargin = 70, rightMargin = 15, verticalMargin = 16;
    const innerWidth = width - leftMargin - rightMargin, innerHeight = height - 2 * verticalMargin;
    const x = time => leftMargin + (time - start) / span * innerWidth;
    const y = value => top === bottom ? height / 2
        : verticalMargin + (top - value / magnitude) / (top - bottom) * innerHeight;

    const tickValues = top === bottom ? [low] : [high, (top / 2 + bottom / 2) * magnitude, low];
    context.font = "12px system-ui, sans-serif";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.lineWidth = 1;
    for (const value of tickValues) {
        const row = y(value);
        context.strokeStyle = "#d5dfe9";
        context.beginPath();
        context.moveTo(leftMargin, row);
        context.lineTo(width - rightMargin, row);
        context.stroke();
        context.fillStyle = "#465970";
        context.fillText(value.toPrecision(4), leftMargin - 7, row);
    }
    context.beginPath();
    context.moveTo(x(start), y(series.values[0]));
    for (let index = 1; index < scene.times.length; index++) {
        context.lineTo(x(scene.times[index]), y(series.values[index]));
    }
    context.strokeStyle = "#173e69";
    context.lineWidth = 2;
    context.stroke();
    context.beginPath();
    context.arc(x(series.peak_time), y(series.values[peakIndex]), 4, 0, 2 * Math.PI);
    context.fillStyle = "#b84613";
    context.fill();
}

nodeSelect.addEventListener("change", selectNode);
componentSelect.addEventListener("change", selectComponent);
sample.addEventListener("input", showSample);
selectNode();
new ResizeObserver(draw).observe(canvas);
