const fs = require("fs");
const path = require("path");
const matter = require("gray-matter");

const directories = [
  { dir: "blog/_posts", output: "blog/index.json" },
  { dir: "note/_posts", output: "note/index.json" }
];

function generateIndex(dir, output) {
  const outputDir = path.dirname(output);

  if (!fs.existsSync(outputDir)) {
    throw new Error(`Directory not found: ${outputDir}`)
  }

  const files = fs.readdirSync(dir).filter(file => file.endsWith(".md"));
  if (files.length === 0) {
    throw new Error(`No markdown files found in: ${dir}`)
  }

  const index = files.map(file => {
    const filePath = path.join(dir, file);
    const content = fs.readFileSync(filePath, "utf8");
    const { data } = matter(content);
    
    return {
      title: data.title || "",
      subtitle: data.subtitle || "",
      description: data.description || "",
      date: data.date || "",
      tags: data.tags || [],
      image: data.image || "",
      link: data.link || file.replace(".md", ""),
    };
  });

  fs.writeFileSync(output, JSON.stringify(index, null, 2));
  console.log(`✅ Index generated: ${output}`);
}

// Generate index for both blog and note sections
directories.forEach(({ dir, output }) => {
   try {
    if (fs.existsSync(dir)) {
      generateIndex(dir, output);
    } else {
      throw new Error(`Directory not found: ${dir}`);
    }
  } catch (err) {
    console.error(`❌ Error generating index for ${dir}: ${err.message}`);
    process.exit(1);
  }
});

