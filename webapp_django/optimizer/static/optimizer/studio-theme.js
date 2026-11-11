/* Appearance is local to the browser; it never changes task or model settings. */
(() => {
  let theme = 'light';
  try { theme = localStorage.getItem('studio-appearance') === 'dark' ? 'dark' : 'light'; } catch (_) {}
  document.documentElement.dataset.theme = theme;
  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.studio-theme-toggle');
    const sync = () => {
      if (!toggle) return;
      const dark = document.documentElement.dataset.theme === 'dark';
      toggle.textContent = dark ? 'Light appearance' : 'Dark appearance';
      toggle.setAttribute('aria-pressed', String(dark));
      toggle.setAttribute('aria-label', dark ? 'Switch to light appearance' : 'Switch to dark appearance');
    };
    if (toggle) toggle.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('studio-appearance', next); } catch (_) {}
      sync();
    });
    sync();
    const titles = {optimize:'Make every prompt count.',ab:'Compare. Learn. Improve.',standings:'Know your models.',history:'Your work, in perspective.',minime:'Teach your own judge.'};
    document.querySelectorAll('.side button[data-view]').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.side button[data-view]').forEach(item => item.removeAttribute('aria-current'));
        button.setAttribute('aria-current','page');
        const title = document.querySelector('#studioTitle');
        if (title) title.textContent = titles[button.dataset.view] || 'Prompt Studio';
      });
    });
    document.querySelectorAll('.door, .stepper .st').forEach(item => {
      item.setAttribute('role','button');
      if (item.matches('.door')) item.setAttribute('aria-pressed', String(item.classList.contains('active')));
      item.tabIndex = 0;
      item.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          if (item.getAttribute('aria-disabled') !== 'true') item.click();
        }
      });
    });
  });
})();
