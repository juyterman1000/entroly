const wasm = require("entroly-wasm");
const adapters = require("./adapters");

module.exports = {
  ...wasm,
  ...adapters,
};
