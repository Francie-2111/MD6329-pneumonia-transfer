const revealTargets = document.querySelectorAll(
  ".section-heading, .question-grid, .pipeline, .method-note, .metric-grid, .feature-figure, .evidence-grid, .closing"
);

if ("IntersectionObserver" in window) {
  revealTargets.forEach((element) => element.classList.add("reveal"));
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  revealTargets.forEach((element) => observer.observe(element));
}
