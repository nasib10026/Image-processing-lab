let time = 0;
let x = [];
let y = [];
let fourierY;
let fourierX;
let path = [];
const TWO_PI = 2 * Math.PI;

function dft(x) {
    let X = [];
    const N = x.length;
    for (let k = 0; k < N; k++) {
        let re = 0;
        let im = 0;
        for (let n = 0; n < N; n++) {
            let phi = (TWO_PI * k * n) / N;
            re += x[n] * Math.cos(phi);
            im += x[n] * Math.sin(phi);
        }
        re = re / N;
        im = im / N;
        let freq = k;
        let amp = Math.sqrt(re * re + im * im);
        let phase = Math.atan2(im, re);
        X[k] = {
            re,
            im,
            freq,
            amp,
            phase
        };
    }
    return X;
}

function epiCycles(x, y, rotation, fourier) {
    let newX = 0;
    let newY = 0;
   for (let i = 0; i < fourier.length; i++) {
        let prevX = x;
        let prevY = y;
        let freq = fourier[i].freq;
        let radius = fourier[i].amp;
        let phase = fourier[i].phase;
        x += radius * Math.cos(freq * time + phase + rotation);
        y += radius * Math.sin(freq * time + phase + rotation);

        stroke(255, 100);
        noFill();
        ellipse(prevX, prevY, radius * 2);
        stroke(255);
        line(prevX, prevY, x, y);
    }

    return createVector(x, y);
}


function setup() {
    createCanvas(600, 400);
    for (let i = 0; i < 100; i++) {
        //let angle = map(i, 0, 100, 0, TWO_PI); // Corrected: Declare angle using let
        x[i] = 50 * noise(i/10);
    }
    for (let i = 0; i < 100; i++) {
        //let angle = map(i, 0, 100, 0, TWO_PI); // Corrected: Declare angle using let
        y[i] = 50 * noise(i/10 + 1000);
    }
    fourierY = dft(y);
    fourierX = dft(x);
}

function draw() {
    background(0);
    let vx = epiCycles(400, 50, 0, fourierX);
    let vy = epiCycles(50, 200, Math.PI, fourierY);
    let v = createVector(vx.x, vy.y);
    path.unshift(v);
    line(vx.x,vx.y,v.x,v.y);
    line(vy.x,vy.y,v.x,v.y);
    beginShape();
    noFill();
    for (let i = 0; i < path.length; i++) {
        vertex(path[i].x, path[i].y);
    }
    endShape();
    const dt = TWO_PI / fourierY.length;
    time += dt;
}
