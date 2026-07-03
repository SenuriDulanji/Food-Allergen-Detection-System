"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, X, ScanLine, AlertTriangle, CheckCircle2, ChevronDown,
  ChevronUp, Info, Zap, Shield, Brain, Loader2, ImageIcon,
} from "lucide-react";
import { scanDish, listAllergens } from "@/lib/api";
import type { ScanResponse, AllergenMatch, ClinicalAlert, SafetyNetWarning } from "@/lib/types";

// ─── Confidence badge ────────────────────────────────────────────────────────
function ConfidenceBadge({ level }: { level: string }) {
  const styles: Record<string, string> = {
    high: "chip-success",
    medium: "chip-warning",
    low: "chip-danger",
  };
  return <span className={`chip ${styles[level] ?? "chip-neutral"}`}>{level} confidence</span>;
}

// ─── Source badge ────────────────────────────────────────────────────────────
function SourceBadge({ source }: { source: string }) {
  const parts = source.split("+");
  const colours: Record<string, string> = {
    rule_based: "chip-info",
    llm: "chip-warning",
    user_profile: "chip-danger",
  };
  return (
    <div className="flex flex-wrap gap-1">
      {parts.map((p) => (
        <span key={p} className={`chip ${colours[p] ?? "chip-neutral"}`}>
          {p.replace("_", " ")}
        </span>
      ))}
    </div>
  );
}

// ─── Scan Results component ──────────────────────────────────────────────────
function ScanResults({ result }: { result: ScanResponse }) {
  const [showIngredients, setShowIngredients] = useState(false);
  const [showML, setShowML] = useState(true);

  const safetyColor = result.is_safe
    ? "border-emerald-500/30 bg-emerald-500/5"
    : "border-red-500/30 bg-red-500/5";

  return (
    <div className="space-y-8 fade-in">
      {/* Safety Banner */}
      <div className={`glass rounded-2xl p-6 border ${safetyColor}`}>
        <div className="flex items-start gap-4">
          <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${result.is_safe ? "bg-emerald-500/20" : "bg-red-500/20"}`}>
            {result.is_safe
              ? <CheckCircle2 size={24} className="text-emerald-400" />
              : <AlertTriangle size={24} className="text-red-400" />}
          </div>
          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-3 mb-2">
              <h2 className="text-xl font-bold capitalize">
                {result.identified_dish.replace(/_/g, " ")}
              </h2>
              <ConfidenceBadge level={result.confidence} />
              {result.rag_context_used && (
                <span className="chip chip-info">
                  <Zap size={10} /> RAG Enhanced
                </span>
              )}
            </div>
            <p className={`font-semibold mb-2 ${result.is_safe ? "text-emerald-400" : "text-red-400"}`}>
              {result.safety_message}
            </p>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              {result.llm_explanation}
            </p>
          </div>
        </div>
      </div>

      {/* Detected Allergens */}
      {result.detected_allergens.length > 0 && (
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-white/5 flex items-center gap-2">
            <AlertTriangle size={16} className="text-[var(--gold)]" />
            <h3 className="font-semibold">
              Detected Allergens
              <span className="ml-2 chip chip-warning">{result.detected_allergens.length}</span>
            </h3>
          </div>
          <div className="divide-y divide-white/5">
            {result.detected_allergens.map((a: AllergenMatch) => (
              <div key={a.allergen_category} className="px-6 py-4 flex flex-wrap items-start gap-3">
                <div className="flex-1 min-w-0">
                  <p className="font-medium capitalize mb-1">
                    {a.allergen_category.replace(/_/g, " ")}
                  </p>
                  <p className="text-xs text-[var(--text-muted)]">
                    Triggered by: {a.triggered_by.join(", ")}
                  </p>
                </div>
                <SourceBadge source={a.source} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ML Risk Report */}
      <div className="glass rounded-2xl overflow-hidden">
        <button
          onClick={() => setShowML(!showML)}
          className="w-full px-6 py-4 border-b border-white/5 flex items-center justify-between hover:bg-white/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Brain size={16} className="text-violet-400" />
            <h3 className="font-semibold">ML Risk Report</h3>
            {result.ml_risk_report.clinical_alerts.length > 0 && (
              <span className="chip chip-danger">{result.ml_risk_report.clinical_alerts.length} alerts</span>
            )}
            {result.ml_risk_report.safety_net_warnings.length > 0 && (
              <span className="chip chip-warning">{result.ml_risk_report.safety_net_warnings.length} warnings</span>
            )}
          </div>
          {showML ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showML && (
          <div className="p-6 space-y-8">
            {/* Stats row */}
            <div className="grid grid-cols-2 gap-3">
              <div className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold gradient-text">{result.ml_risk_report.models_evaluated}</div>
                <div className="text-xs text-[var(--text-muted)] mt-1">Models Evaluated</div>
              </div>
              <div className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold gradient-text">{result.ml_risk_report.ingredients_checked}</div>
                <div className="text-xs text-[var(--text-muted)] mt-1">Ingredients Checked</div>
              </div>
            </div>

            {/* Clinical alerts */}
            {result.ml_risk_report.clinical_alerts.length > 0 ? (
              <div>
                <h4 className="text-sm font-semibold text-[var(--text-secondary)] mb-3 flex items-center gap-1.5">
                  <Brain size={13} className="text-violet-400" /> Clinical Alerts (XGBoost)
                </h4>
                <div className="space-y-2">
                  {result.ml_risk_report.clinical_alerts.map((a: ClinicalAlert) => (
                    <div key={a.ingredient} className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-red-300 capitalize">{a.ingredient}</span>
                        <div className="flex items-center gap-2">
                          <span className="chip chip-danger text-xs">Risk {a.risk_level}</span>
                          <span className="chip chip-neutral text-xs">{a.confidence}</span>
                        </div>
                      </div>
                      <p className="text-xs text-[var(--text-muted)]">{a.msg}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] rounded-xl border border-white/5 p-4">
                <Info size={14} />
                No clinical risk alerts from ML models for this profile.
              </div>
            )}

            {/* Safety net warnings */}
            {result.ml_risk_report.safety_net_warnings.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-[var(--text-secondary)] mb-3 flex items-center gap-1.5">
                  <Shield size={13} className="text-amber-400" /> Cross-Reactivity Safety Net (Apriori)
                </h4>
                <div className="space-y-2">
                  {result.ml_risk_report.safety_net_warnings.map((w: SafetyNetWarning) => (
                    <div key={w.trigger} className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <span className="font-medium text-amber-300 capitalize">{w.trigger}</span>
                        <span className="text-[var(--text-muted)] text-xs">→ also reactive with:</span>
                        {w.linked_risks.map((r) => (
                          <span key={r} className="chip chip-warning text-xs">{r}</span>
                        ))}
                      </div>
                      <p className="text-xs text-[var(--text-muted)]">{w.msg}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Ingredients */}
      <div className="glass rounded-2xl overflow-hidden">
        <button
          onClick={() => setShowIngredients(!showIngredients)}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors"
        >
          <h3 className="font-semibold flex items-center gap-2">
            Ingredients
            <span className="chip chip-neutral">{result.ingredients.length}</span>
          </h3>
          {showIngredients ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {showIngredients && (
          <div className="px-6 pb-5 flex flex-wrap gap-2">
            {result.ingredients.map((ing) => (
              <span key={ing} className="chip chip-neutral">{ing}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Scan Page ───────────────────────────────────────────────────────────
export default function ScanPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [allergenOptions, setAllergenOptions] = useState<string[]>([]);
  const [selectedAllergens, setSelectedAllergens] = useState<string[]>([]);
  const [userId, setUserId] = useState<string>("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listAllergens().then(setAllergenOptions).catch(() => {});
    // Load saved userId from localStorage
    const saved = localStorage.getItem("allerscan_user_id");
    if (saved) setUserId(saved);
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, or WEBP).");
      return;
    }
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const toggleAllergen = (a: string) => {
    setSelectedAllergens((prev) =>
      prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]
    );
  };

  const handleScan = async () => {
    if (!imageFile) return;
    setScanning(true);
    setError(null);
    setResult(null);
    try {
      const uid = userId.trim() ? parseInt(userId, 10) : undefined;
      const data = await scanDish(imageFile, selectedAllergens, uid);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Scan failed. Please try again.");
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-4xl font-extrabold mb-2">
          <span className="gradient-text">Scan</span> Your Dish
        </h1>
        <p className="text-[var(--text-secondary)]">
          Upload a photo of any Sri Lankan dish and get an instant 4-layer allergen analysis.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* ── Left: Upload + Config ─────────────────────────────────── */}
        <div className="space-y-8">

          {/* Drop zone */}
          <div
            className={`drop-zone rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer min-h-[280px] relative transition-all ${isDragging ? "drag-over" : ""}`}
            onDragEnter={() => setIsDragging(true)}
            onDragLeave={() => setIsDragging(false)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {imagePreview ? (
              <div className="relative w-full h-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imagePreview}
                  alt="Dish preview"
                  className="w-full h-56 object-cover rounded-xl"
                />
                <button
                  className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/60 flex items-center justify-center hover:bg-red-500/80 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    setImageFile(null);
                    setImagePreview(null);
                    setResult(null);
                  }}
                >
                  <X size={14} className="text-white" />
                </button>
                <p className="text-center text-sm text-[var(--text-muted)] mt-3">
                  {imageFile?.name}
                </p>
              </div>
            ) : (
              <>
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#FF6B35]/20 to-[#F59E0B]/10 flex items-center justify-center mb-4">
                  <ImageIcon size={26} className="text-[var(--saffron)]" />
                </div>
                <p className="font-semibold text-[var(--text-primary)] mb-1">Drop your dish photo here</p>
                <p className="text-sm text-[var(--text-muted)]">or click to browse</p>
                <p className="text-xs text-[var(--text-muted)] mt-3">JPEG · PNG · WEBP · max 10 MB</p>
              </>
            )}
          </div>

          {/* User ID field */}
          <div className="glass rounded-2xl p-5">
            <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
              User ID <span className="text-[var(--text-muted)]">(optional — loads saved profile)</span>
            </label>
            <input
              type="number"
              placeholder="e.g. 1"
              value={userId}
              onChange={(e) => {
                setUserId(e.target.value);
                if (e.target.value) localStorage.setItem("allerscan_user_id", e.target.value);
              }}
              className="input-field w-full px-4 py-2.5 rounded-xl text-sm"
            />
          </div>

          {/* Allergen picker */}
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-medium text-[var(--text-secondary)]">
                My Allergens
              </label>
              {selectedAllergens.length > 0 && (
                <button
                  className="text-xs text-[var(--text-muted)] hover:text-[var(--saffron)] transition-colors"
                  onClick={() => setSelectedAllergens([])}
                >
                  Clear all
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {allergenOptions.map((a) => (
                <button
                  key={a}
                  onClick={() => toggleAllergen(a)}
                  className={`allergen-toggle px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all ${
                    selectedAllergens.includes(a) ? "selected" : ""
                  }`}
                >
                  {a.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>

          {/* Scan button */}
          <button
            onClick={handleScan}
            disabled={!imageFile || scanning}
            className="btn-primary w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 text-base"
          >
            {scanning ? (
              <>
                <Loader2 size={18} className="spin" />
                Analysing dish…
              </>
            ) : (
              <>
                <ScanLine size={18} />
                Scan for Allergens
              </>
            )}
          </button>

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300 flex items-start gap-2">
              <AlertTriangle size={16} className="shrink-0 mt-0.5" />
              {error}
            </div>
          )}
        </div>

        {/* ── Right: Results ────────────────────────────────────────── */}
        <div>
          {result ? (
            <ScanResults result={result} />
          ) : (
            <div className="glass rounded-2xl h-full min-h-[400px] flex flex-col items-center justify-center text-center p-10 border-2 border-dashed border-white/5">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#FF6B35]/10 to-[#F59E0B]/5 flex items-center justify-center mb-4">
                <ScanLine size={28} className="text-[var(--text-muted)]" />
              </div>
              <p className="font-medium text-[var(--text-secondary)] mb-1">Results appear here</p>
              <p className="text-sm text-[var(--text-muted)]">
                Upload a dish photo and tap Scan to begin analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
