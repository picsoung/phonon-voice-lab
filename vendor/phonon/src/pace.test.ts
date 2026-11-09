import { test } from "node:test";
import assert from "node:assert/strict";
import { paceForSpeed } from "./pace";

test("maps speaking speed to the model's pace, between and beyond the measured points", () => {
  assert.equal(paceForSpeed(1), 0);
  assert.ok(Math.abs(paceForSpeed(1.2) - -0.28) < 0.005);
  assert.equal(paceForSpeed(0.5), 0.6);
  assert.equal(paceForSpeed(2), -0.45);
});
