// Lists for the forms: institutions, hospitals, and approved consultants (name and hospital only) to choose as supervisor.
import { J, ready, currentUser } from '../../server/auth.js';

export const INSTITUTIONS = ['University of Nairobi', 'Moi University', 'Aga Khan University', 'Kenyatta University', 'Egerton University', 'Maseno University',
  'Jomo Kenyatta University of Agriculture and Technology', 'Mount Kenya University', 'Uzima University', 'Kenya Methodist University', 'COSECSA programme', 'Other'];
export const HOSPITALS = ['Kenyatta National Hospital', 'Moi Teaching and Referral Hospital', 'Aga Khan University Hospital, Nairobi', 'The Nairobi Hospital',
  'Kenyatta University Teaching, Referral and Research Hospital', 'Mater Misericordiae Hospital', 'MP Shah Hospital', 'Tenwek Hospital', 'Kijabe Hospital',
  'Coast General Teaching and Referral Hospital', 'Jaramogi Oginga Odinga Teaching and Referral Hospital', 'Other'];

export async function onRequestGet({ request, env }) {
  const out = { institutions: INSTITUTIONS, hospitals: HOSPITALS, consultants: [] };
  if (!env.DB) return J(out);
  await ready(env);
  const u = await currentUser(request, env);
  if (u && u.status === 'active') {
    out.consultants = (await env.DB.prepare("SELECT id, name, hospital FROM users WHERE role = 'consultant' AND status = 'active' ORDER BY name").all()).results;
  }
  return J(out);
}
