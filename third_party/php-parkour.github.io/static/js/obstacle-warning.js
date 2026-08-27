window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin || event.data?.type !== 'php-obstacle-warning') {
    return;
  }

  const isDemoFrame = Array.from(document.querySelectorAll('iframe'))
    .some((frame) => frame.contentWindow === event.source);
  if (!isDemoFrame) {
    return;
  }

  document.querySelectorAll('[data-obstacle-warning]').forEach((warning) => {
    warning.classList.toggle('is-active', event.data.active === true);
  });
});
