const fs = require("fs");
const path = require("path");
const matter = require("gray-matter");

const directories = [
  { dir: "blog/_posts", output: "blog/index.json" },
  { dir: "note/_posts", output: "note/index.json" }
];

function generateIndex(dir, output) {
  const files = fs.readdirSync(dir).filter(file => file.endsWith(".md"));
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
  if (fs.existsSync(dir)) {
    generateIndex(dir, output);
  } else {
    console.warn(`⚠️ Directory not found: ${dir}`);
  }
});

