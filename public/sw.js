// Offline and repeat-visit speed. Pages and the procedure text: network first, so a new deploy shows at once, with
// the cached copy when offline. The app bundle, CT volumes, meshes and textures: from the cache at once, refreshed in
// the background (a changed file appears on the next visit).
const CACHE = 'cova-v5';
const FRESH = (u) => u.pathname === '/' || u.pathname.endsWith('.html') || u.pathname.endsWith('.json') || u.pathname.endsWith('.webmanifest');

self.addEventListener('install', (e) => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then((c) => c.add('/').catch(() => undefined))); });
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request; if (req.method !== 'GET') return;
  const url = new URL(req.url); if (url.origin !== self.location.origin) return;
  // the account API is always live: never cache sign-in state, logbook or plans
  if (url.pathname.startsWith('/api/')) return;
  // video and other ranged requests go straight to the network: a cached whole file cannot answer a byte-range request
  if (req.headers.has('range') || url.pathname.startsWith('/media/') || /\.(mp4|webm|mov)$/.test(url.pathname)) return;
  const key = req.mode === 'navigate' ? '/' : req;
  if (req.mode === 'navigate' || FRESH(url)) {
    e.respondWith(fetch(req).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(CACHE).then((c) => c.put(key, cp)); } return r; })
      .catch(() => caches.match(key).then((hit) => hit || new Response('Offline, and this page has not been opened before.', { status: 503 }))));
    return;
  }
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(req);
    const net = fetch(req).then((r) => { if (r.ok && r.status === 200) c.put(req, r.clone()); return r; });
    if (hit) { e.waitUntil(net.catch(() => undefined)); return hit; }
    return net;
  }));
});
