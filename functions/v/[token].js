// Old one-tap verification links from weekly emails. Cases no longer need consultant sign-off, so these just open COVA.
export const onRequest = ({ request }) => Response.redirect(new URL('/', request.url).toString(), 302);
