"use client";

import { useEffect, useRef } from "react";

/**
 * Embeds the server-generated Plotly HTML inside an isolated iframe so its
 * scripts can run without polluting the React document.
 */
export function Plot3D({ html }: { html: string }) {
  const ref = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const doc = ref.current.contentDocument;
    if (!doc) return;
    doc.open();
    doc.write(`<!doctype html><html><head><meta charset='utf-8'><style>
      html,body{margin:0;padding:0;background:transparent;font-family:Inter,system-ui,sans-serif;}
      .plotly-graph-div{width:100%!important;}
    </style></head><body>${html}</body></html>`);
    doc.close();
  }, [html]);

  return (
    <iframe
      ref={ref}
      title="3D viewer"
      className="h-[720px] w-full rounded-lg border bg-[#0f1117]"
    />
  );
}
