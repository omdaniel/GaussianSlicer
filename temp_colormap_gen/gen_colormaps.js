const d3 = require("d3-scale-chromatic");
const d3Color = require("d3-color");

const colormaps = {
    "VIRIDIS": d3.interpolateViridis,
    "PLASMA": d3.interpolatePlasma,
    "MAGMA": d3.interpolateMagma,
    "INFERNO": d3.interpolateInferno,
    "TURBO": d3.interpolateTurbo,
    "YLORRD": d3.interpolateYlOrRd,
    "CIVIDIS": d3.interpolateCividis,
    "TWILIGHT": d3.interpolateTwilight,
    "SPECTRAL": d3.interpolateSpectral,
    "RDBU": d3.interpolateRdBu,
    "PIYG": d3.interpolatePiYG,
    "CUBEHELIX": d3.interpolateCubehelixDefault,
};

function formatVec3(colorStr) {
    const c = d3Color.color(colorStr).rgb();
    return `vec3<f32>(${(c.r / 255).toFixed(5)}, ${(c.g / 255).toFixed(5)}, ${(c.b / 255).toFixed(5)})`;
}

for (const [name, interpolator] of Object.entries(colormaps)) {
    if (!interpolator) {
        console.log(`// Missing interpolator for ${name}`);
        continue;
    }
    console.log(`const ${name}_LUT: array<vec3<f32>, 256> = array<vec3<f32>, 256>(`);
    for (let i = 0; i < 256; i++) {
        const t = i / 255.0;
        console.log(`    ${formatVec3(interpolator(t))},`);
    }
    console.log(`);`);
    console.log("");
}
