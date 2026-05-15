import { Topbar } from "@/components/topbar";
import { UploadForm } from "@/components/upload-form";
import { HistoryList } from "@/components/history-list";

export default function Home() {
  return (
    <>
      <Topbar />
      <main className="mx-auto max-w-5xl space-y-4 px-4 py-8 md:px-6 md:py-10">
        <header>
          <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
            Phân tích CT phổi
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload ảnh CT, điền thông tin lâm sàng (tuỳ chọn). AI sẽ phát hiện nốt,
            phân loại malignancy, và đánh giá nguy cơ ung thư theo Brock + Lung-RADS.
          </p>
        </header>
        <UploadForm />
        <HistoryList />
      </main>
    </>
  );
}
