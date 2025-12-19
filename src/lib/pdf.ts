import html2canvas from "html2canvas";
import jsPDF from "jspdf";

export async function exportElementToPdf(element: HTMLElement, filename: string) {
  const canvas = await html2canvas(element, { scale: 2, useCORS: true, backgroundColor: "#ffffff" });
  const imgData = canvas.toDataURL("image/png");
  const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });

  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();

  // Keep aspect ratio
  const imgWidth = pageWidth - 72; // margins
  const ratio = canvas.height / canvas.width;
  const imgHeight = imgWidth * ratio;

  let y = 72; // top margin
  let x = 36; // left margin

  if (imgHeight <= pageHeight - 144) {
    pdf.addImage(imgData, "PNG", x, y, imgWidth, imgHeight);
  } else {
    // Multi-page
    let remaining = imgHeight;
    let position = y;
    const sliceHeight = pageHeight - 144; // usable height per page

    while (remaining > 0) {
      pdf.addImage(imgData, "PNG", x, position, imgWidth, imgHeight);
      remaining -= sliceHeight;
      if (remaining > 0) {
        pdf.addPage();
        position = 72 - (imgHeight - remaining); // continue from where we left off
      }
    }
  }

  pdf.save(filename);
}
