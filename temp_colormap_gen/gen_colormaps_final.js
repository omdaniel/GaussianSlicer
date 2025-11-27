const d3 = require("d3-scale-chromatic");
const d3Color = require("d3-color");


const d3_colormaps = {
    "VIRIDIS": d3.interpolateViridis,
    "PLASMA": d3.interpolatePlasma,
    "MAGMA": d3.interpolateMagma,
    "INFERNO": d3.interpolateInferno,
    "TURBO": d3.interpolateTurbo,
    "YLORRD": d3.interpolateYlOrRd,
    "CIVIDIS": d3.interpolateCividis,
    "SINEBOW": d3.interpolateSinebow,
    "SPECTRAL": d3.interpolateSpectral,
    "RDBU": d3.interpolateRdBu,
    "PIYG": d3.interpolatePiYG,
    "CUBEHELIX": d3.interpolateCubehelixDefault,
    "PUOR": d3.interpolatePuOr,
    "BRBG": d3.interpolateBrBG,
};

const custom_luts = {
    "COOLWARM": [
        [0.22980, 0.29872, 0.75368],
        [0.26679, 0.43538, 0.82257],
        [0.30659, 0.57122, 0.87068],
        [0.34981, 0.70415, 0.89244],
        [0.39504, 0.83206, 0.88175],
        [0.75475, 0.90930, 0.74860],
        [0.88245, 0.77634, 0.51465],
        [0.90493, 0.63036, 0.31846],
        [0.89376, 0.47136, 0.16899],
        [0.84433, 0.30354, 0.07442],
        [0.78039, 0.12619, 0.01519]
    ],
    "BLUEORANGE": [
        [0.034, 0.113, 0.345],
        [0.076, 0.295, 0.582],
        [0.156, 0.479, 0.741],
        [0.274, 0.647, 0.818],
        [0.445, 0.791, 0.827],
        [0.659, 0.859, 0.749],
        [0.835, 0.812, 0.588],
        [0.938, 0.682, 0.402],
        [0.964, 0.482, 0.259],
        [0.916, 0.262, 0.170],
        [0.800, 0.063, 0.119]
    ],
    "SEISMIC": [
        [0.000, 0.000, 0.300],
        [0.000, 0.000, 0.700],
        [0.000, 0.400, 1.000],
        [0.200, 0.700, 1.000],
        [0.500, 0.900, 1.000],
        [0.800, 0.800, 0.800],
        [1.000, 0.600, 0.600],
        [1.000, 0.300, 0.200],
        [0.900, 0.000, 0.000],
        [0.600, 0.000, 0.000],
        [0.300, 0.000, 0.000]
    ],
    "HOT": [
        [0.00000, 0.00000, 0.00000],
        [0.20000, 0.00000, 0.00000],
        [0.40000, 0.00000, 0.00000],
        [0.60000, 0.00000, 0.00000],
        [0.80000, 0.00000, 0.00000],
        [1.00000, 0.20000, 0.00000],
        [1.00000, 0.40000, 0.00000],
        [1.00000, 0.60000, 0.00000],
        [1.00000, 0.80000, 0.00000],
        [1.00000, 0.90000, 0.20000],
        [1.00000, 1.00000, 1.00000]
    ]
};

function formatVec3(r, g, b) {
    return `vec3<f32>(${r.toFixed(5)}, ${g.toFixed(5)}, ${b.toFixed(5)})`;
}

function interpolateCustom(points, t) {
    const n = points.length - 1;
    const scaled = t * n;
    let i = Math.floor(scaled);
    if (i >= n) i = n - 1;
    if (i < 0) i = 0;
    const f = scaled - i;

    const c0 = points[i];
    const c1 = points[i + 1];

    return [
        c0[0] * (1 - f) + c1[0] * f,
        c0[1] * (1 - f) + c1[1] * f,
        c0[2] * (1 - f) + c1[2] * f
    ];
}

// D3 Maps
for (const [name, interpolator] of Object.entries(d3_colormaps)) {
    if (typeof interpolator !== 'function') {
        console.error(`Error: ${name} interpolator is not a function:`, interpolator);
        continue;
    }
    console.log(`const ${name}_LUT: array<vec3<f32>, 256> = array<vec3<f32>, 256>(`);
    for (let i = 0; i < 256; i++) {
        const t = i / 255.0;
        const c = d3Color.color(interpolator(t)).rgb();
        console.log(`    ${formatVec3(c.r / 255, c.g / 255, c.b / 255)},`);
    }
    console.log(`);`);
    console.log("");
}

// Custom Maps
for (const [name, points] of Object.entries(custom_luts)) {
    console.log(`const ${name}_LUT: array<vec3<f32>, 256> = array<vec3<f32>, 256>(`);
    for (let i = 0; i < 256; i++) {
        const t = i / 255.0;
        const c = interpolateCustom(points, t);
        console.log(`    ${formatVec3(c[0], c[1], c[2])},`);
    }
    console.log(`);`);
    console.log("");
}
