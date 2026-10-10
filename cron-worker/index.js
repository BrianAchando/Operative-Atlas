export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(fetch(`${env.SITE}/api/digest`, { method: 'POST', headers: { 'x-cron-key': env.CRON_KEY ?? '' } })
      .then(async (r) => console.log('digest', r.status, await r.text())));
  },
};
