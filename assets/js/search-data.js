// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-beyond-static-embeddings-training-a-6-000-parameter-rl-agent-to-actively-hunt-down-historical-planning-maps",
        
          title: "Beyond Static Embeddings: Training a 6,000-Parameter RL Agent to Actively Hunt Down Historical...",
        
        description: "Matching planning documents to the entirety of Greater Manchester using tiny models and reinforcemment learning",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/osmapping-rl/";
          
        },
      },{id: "post-sovereign-ai-for-greater-manchester-geolocating-20-years-of-planning-docs-with-edge-ready-models",
        
          title: "Sovereign AI for Greater Manchester: Geolocating 20 Years of Planning Docs with Edge-Ready...",
        
        description: "Matching planning documents to the entirety of Greater Manchester using contrastive learning and synthetic data.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/sovereign-ai-manchester/";
          
        },
      },{id: "post-building-a-gaia-1-style-world-model-for-indoor-robot-navigation",
        
          title: "Building a GAIA-1 Style World Model for Indoor Robot Navigation",
        
        description: "How I trained an action-conditioned video world model from scratch on a home robot dataset.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/gaia-world-model/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
