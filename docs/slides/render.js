import fs from "fs";
import { marked } from "marked";

const mdFile = process.argv[2];
const templateFile = process.argv[3];
const outFile = process.argv[4];
const title = process.argv[5] || "Slides";
const classNo = process.argv[6] || "";
const classTitle = process.argv[7] || "";

const md = fs.readFileSync(mdFile, "utf8");

// Reveal-style slide splitting
const slides = md
  .split(/^---$/m)
  .map(block => {
    const vertical = block.split(/^--$/m);
    if (vertical.length === 1) {
      return `<section>${marked.parse(block)}</section>`;
    }
    return `<section>${vertical
      .map(v => `<section>${marked.parse(v)}</section>`)
      .join("")}</section>`;
  })
  .join("\n");

// Read template and substitute placeholders
let html = fs.readFileSync(templateFile, "utf8");
html = html.replace("{{SLIDES_HTML}}", slides);
html = html.replace("{{TITLE}}", title);
html = html.replace("{{CLASS_NO}}", classNo);
html = html.replace("{{CLASS_TITLE}}", classTitle);

fs.writeFileSync(outFile, html);
