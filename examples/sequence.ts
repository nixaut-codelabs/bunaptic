import { Architect } from "../src/index.js";

const model = Architect.LSTM(1, 8, 1);

const short = new Float32Array([1, 2, 3, 4, 5]);
const long = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

console.log("short steps", model.runSequence(short).length);
console.log("long steps", model.runSequence(long).length);
