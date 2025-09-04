const CACHE_NAME = 'ina-pwa-v2';
const ASSETS = ['/', '/webpage.html', '/manifest.webmanifest'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((c) => c.addAll(ASSETS)));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  const isAPI = url.pathname.startsWith('/api/');

  // 关键：非 GET 的 API 请求（POST/PUT/DELETE…）不缓存，直接走网络
  if (isAPI && e.request.method !== 'GET') {
    e.respondWith(fetch(e.request));
    return;
  }

  if (isAPI) {
    // GET 的 API：网络优先，成功后异步写缓存
    e.respondWith(
      fetch(e.request).then((r) => {
        caches.open(CACHE_NAME).then((c) => c.put(e.request, r.clone())).catch(() => {});
        return r;
      }).catch(() => caches.match(e.request))
    );
    return;
  }

  // 其他静态资源：缓存优先
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
