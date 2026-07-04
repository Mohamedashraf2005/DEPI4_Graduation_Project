import React, { useRef, useState } from 'react';

export default function ImageUploader() {
  // 1. References & State
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // 2. The function that runs when you select a file
  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Show preview of the selected image
    setSelectedImage(URL.createObjectURL(file));
    setIsUploading(true);
    setResults(null);

    // 3. Prepare the FormData exactly how FastAPI expects it
    const formData = new FormData();
    formData.append('file', file); // 'file' must match the parameter name in your FastAPI app.py

    try {
      // 4. Send to both 8001 (Accident) and 8002 (Traffic Sign) concurrently!
      const [accidentRes, trafficRes] = await Promise.all([
        fetch('http://localhost:8001/predict', { method: 'POST', body: formData }),
        fetch('http://localhost:8002/predict', { method: 'POST', body: formData })
      ]);

      const accidentData = await accidentRes.json();
      const trafficData = await trafficRes.json();

      // 5. Save the combined results to state to display them
      setResults({
        accidents: accidentData,
        traffic: trafficData
      });

    } catch (error) {
      console.error("Error connecting to APIs:", error);
      alert("حدث خطأ في الاتصال بالسيرفر. تأكد من تشغيل 8001 و 8002.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4" dir="rtl">
      
      {/* Hidden File Input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/jpeg, image/png, video/mp4" 
        className="hidden" 
      />

      {/* Your exact UI Button, now clickable! */}
      <button 
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        className="group flex w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-panel/50 px-6 py-16 text-center transition hover:border-primary/50 hover:bg-primary/[0.03] disabled:opacity-50"
      >
        <span className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary transition group-hover:scale-105">
          {isUploading ? (
            // Simple loading spinner
            <div className="h-7 w-7 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
          ) : (
            // Your Upload Icon
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-cloud-upload h-7 w-7"><path d="M12 13v8"></path><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path><path d="m8 17 4-4 4 4"></path></svg>
          )}
        </span>
        <div className="mt-4 font-semibold text-ink">
          {isUploading ? 'جاري التحليل...' : 'اسحب الملف هنا أو اضغط للاختيار'}
        </div>
        <div className="mt-1 text-xs text-ink-faint">صور JPG/PNG أو فيديو MP4</div>
      </button>

      {/* Display Results */}
      {results && (
        <div className="mt-8 p-6 bg-white rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-bold mb-4">نتائج التحليل:</h3>
          
          {selectedImage && (
            <img src={selectedImage} alt="Uploaded" className="w-full max-h-64 object-cover rounded-lg mb-4" />
          )}

          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-red-50 rounded-lg">
              <h4 className="font-bold text-red-700">حوادث السيارات (8001)</h4>
              <pre className="mt-2 text-sm text-left dir-ltr overflow-auto">
                {JSON.stringify(results.accidents, null, 2)}
              </pre>
            </div>
            
            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-bold text-blue-700">العلامات المرورية (8002)</h4>
              <pre className="mt-2 text-sm text-left dir-ltr overflow-auto">
                {JSON.stringify(results.traffic, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}