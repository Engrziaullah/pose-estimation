/**
 * initDropzone
 * Wires a <label class="dropzone"> + hidden <input type="file"> pair so it:
 *  - opens the file picker on click (native <label for> behavior)
 *  - accepts drag-and-drop
 *  - shows the chosen filename
 * Submission itself is handled per-page via fetch(), so this only owns
 * file selection UX.
 */
function initDropzone(dropzoneId, inputId, filenameId) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  const filenameEl = document.getElementById(filenameId);
  if (!dropzone || !input) return;

  const defaultText = filenameEl.textContent;

  const showName = (name) => {
    filenameEl.textContent = name;
    dropzone.classList.add('has-file');
  };

  input.addEventListener('change', () => {
    if (input.files && input.files[0]) showName(input.files[0].name);
  });

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('is-dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('is-dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      input.files = files;
      showName(files[0].name);
    }
  });
}

/**
 * setBusy
 * Toggles a submit button between idle and busy (spinner + label swap),
 * shared by the image and video AJAX flows.
 */
function setBusy(btn, busy, label) {
  const labelEl = btn.querySelector('.btn-label');
  const spinnerEl = btn.querySelector('.btn-spinner');
  btn.disabled = busy;
  if (labelEl && label) labelEl.textContent = label;
  if (spinnerEl) spinnerEl.hidden = !busy;
}

/* ---------------------------------------------------------------------- */
/* Chrome: sticky nav shadow, mobile menu, reveal-on-load animations       */
/* ---------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  const topbar = document.getElementById('topbar');
  if (topbar) {
    const onScroll = () => topbar.classList.toggle('is-scrolled', window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  const navToggle = document.getElementById('nav-toggle');
  const topnav = document.getElementById('topnav');
  if (navToggle && topnav) {
    navToggle.addEventListener('click', () => {
      const open = topnav.classList.toggle('is-open');
      navToggle.classList.toggle('is-open', open);
      navToggle.setAttribute('aria-expanded', String(open));
    });
    topnav.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
      topnav.classList.remove('is-open');
      navToggle.classList.remove('is-open');
    }));
  }

  const revealables = document.querySelectorAll('.reveal');
  revealables.forEach((el, i) => {
    el.style.setProperty('--reveal-delay', `${i * 70}ms`);
    requestAnimationFrame(() => el.classList.add('is-visible'));
  });
});
