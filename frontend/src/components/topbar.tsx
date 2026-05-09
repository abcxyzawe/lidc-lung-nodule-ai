import Link from "next/link";

export function Topbar({ subtitle }: { subtitle?: string }) {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 md:px-6">
        <Link href="/" className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-blue-700 to-teal-600 text-white shadow-sm">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              className="h-5 w-5"
            >
              <circle cx="12" cy="12" r="9" strokeDasharray="2 2" />
              <circle cx="12" cy="12" r="5.5" />
              <circle cx="12" cy="12" r="2" fill="currentColor" />
            </svg>
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-bold tracking-tight">Nodura</div>
            <div className="text-[11px] text-muted-foreground">
              {subtitle ?? "Soi rõ từng nốt, an tâm từng hơi thở"}
            </div>
          </div>
        </Link>

        <div className="hidden items-center gap-5 text-xs text-muted-foreground md:flex">
          <span>
            Test Dice <b className="font-semibold text-foreground">0.7502</b>
          </span>
          <span>
            Recall <b className="font-semibold text-foreground">97.5%</b>
          </span>
          <span>
            Brock val <b className="font-semibold text-foreground">0.46</b>
          </span>
        </div>
      </div>
    </header>
  );
}
