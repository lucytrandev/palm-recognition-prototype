// Upload a palm image: read it as a data URI and POST it; the results panel updates live.
document.addEventListener('change', (e) => {
  if (!e.target.matches || !e.target.matches('#palm-file')) return;
  const f = e.target.files[0];
  if (!f) return;
  const note = document.getElementById('upload-note');
  if (note) note.textContent = 'matching ' + f.name + '…';
  const r = new FileReader();
  r.onload = async () => {
    try {
      const res = await fetch('/api/match', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({file: f.name, image: r.result})});
      if (note) note.textContent = res.ok ? '' : 'error ' + res.status;
    } catch (err) { if (note) note.textContent = String(err); }
    e.target.value = '';
  };
  r.readAsDataURL(f);
});
