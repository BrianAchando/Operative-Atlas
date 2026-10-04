import { onRequestGet as __api_flags_js_onRequestGet } from "C:\\Users\\mason\\hilum\\functions\\api\\flags.js"
import { onRequestPatch as __api_flags_js_onRequestPatch } from "C:\\Users\\mason\\hilum\\functions\\api\\flags.js"
import { onRequestPost as __api_flags_js_onRequestPost } from "C:\\Users\\mason\\hilum\\functions\\api\\flags.js"

export const routes = [
    {
      routePath: "/api/flags",
      mountPath: "/api",
      method: "GET",
      middlewares: [],
      modules: [__api_flags_js_onRequestGet],
    },
  {
      routePath: "/api/flags",
      mountPath: "/api",
      method: "PATCH",
      middlewares: [],
      modules: [__api_flags_js_onRequestPatch],
    },
  {
      routePath: "/api/flags",
      mountPath: "/api",
      method: "POST",
      middlewares: [],
      modules: [__api_flags_js_onRequestPost],
    },
  ]