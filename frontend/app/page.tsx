import Link from "next/link";
import { ScanLine, Brain, Shield, Zap, ChevronRight, Star } from "lucide-react";

const FEATURES = [
  {
    icon: ScanLine,
    color: "from-[#FF6B35] to-[#F59E0B]",
    title: "AI Vision Analysis",
    desc: "Gemini 2.5 Flash identifies Sri Lankan dishes from a single photo with remarkable accuracy.",
  },
  {
    icon: Brain,
    color: "from-[#8B5CF6] to-[#60A5FA]",
    title: "ML Risk Prediction",
    desc: "XGBoost + SMOTE models trained on real survey data predict your personalised clinical allergy risk.",
  },
  {
    icon: Shield,
    color: "from-[#10B981] to-[#34D399]",
    title: "Apriori Safety Net",
    desc: "Validated association rules flag cross-reactive allergens — the invisible risks most systems miss.",
  },
  {
    icon: Zap,
    color: "from-[#F59E0B] to-[#FCD34D]",
    title: "RAG Knowledge Base",
    desc: "ChromaDB retrieval gives the AI deep dish-specific context for every scan — zero hallucinations.",
  },
];

const DISHES = [
  "🍛 Fish Curry", "🥥 Coconut Sambol", "🦐 Prawn Curry",
  "🍚 Biriyani", "🌶️ Pol Roti", "🥘 Dhal Curry",
];

const STATS = [
  { label: "Allergens Tracked", value: "24" },
  { label: "Sri Lankan Dishes", value: "150+" },
  { label: "ML Models", value: "24" },
  { label: "Detection Layers", value: "4" },
];

export default function HomePage() {
  return (
    <div className="max-w-6xl mx-auto px-6">

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="pt-20 pb-24 text-center relative">
        {/* Background glow orb */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[600px] h-[600px] rounded-full bg-[var(--saffron)] opacity-[0.04] blur-[120px]" />
        </div>

        <div className="relative">
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full glass border border-[rgba(255,107,53,0.2)] text-sm text-[var(--saffron-light)]">
            <span className="pulse-dot green" />
            <span>4-Layer AI Detection System · Powered by Gemini 2.5 Flash</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.1] mb-6">
            <span className="text-[var(--text-primary)]">Know What's</span>
            <span className="gradient-text">In Your Plate.</span>
          </h1>

          <p className="text-lg md:text-xl text-[var(--text-secondary)] max-w-2xl mx-auto mb-10 leading-relaxed">
            Snap a photo of any Sri Lankan dish. Our AI instantly detects allergens,
            predicts your personal clinical risk, and warns you about hidden cross-reactivities —
            so you can eat confidently.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/scan"
              className="btn-primary px-8 py-4 rounded-xl text-base font-semibold flex items-center gap-2 w-full sm:w-auto justify-center"
            >
              <ScanLine size={18} />
              Start Scanning Free
              <ChevronRight size={16} />
            </Link>
            <Link
              href="/profile"
              className="btn-ghost px-8 py-4 rounded-xl text-base font-medium flex items-center gap-2 w-full sm:w-auto justify-center"
            >
              Create Profile
            </Link>
          </div>

          {/* Dish pills */}
          <div className="flex flex-wrap justify-center gap-2 mt-10">
            {DISHES.map((dish) => (
              <span key={dish} className="chip chip-neutral text-sm py-1.5 px-3">
                {dish}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Stats ──────────────────────────────────────────────────────── */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-20">
        {STATS.map(({ label, value }) => (
          <div key={label} className="glass rounded-2xl p-6 text-center glow-saffron">
            <div className="text-3xl font-extrabold gradient-text mb-1">{value}</div>
            <div className="text-sm text-[var(--text-secondary)]">{label}</div>
          </div>
        ))}
      </section>

      {/* ── Features ───────────────────────────────────────────────────── */}
      <section className="mb-24">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-3">
            Four Layers of Protection
          </h2>
          <p className="text-[var(--text-secondary)] max-w-xl mx-auto">
            A hybrid pipeline that combines vision AI, machine learning, and association mining
            — all working together to keep you safe.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-5">
          {FEATURES.map(({ icon: Icon, color, title, desc }) => (
            <div
              key={title}
              className="glass rounded-2xl p-6 flex gap-5 hover:border-[rgba(255,107,53,0.2)] transition-all group"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center shrink-0 shadow-lg group-hover:scale-110 transition-transform`}>
                <Icon size={22} className="text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-[var(--text-primary)] mb-1">{title}</h3>
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA Banner ─────────────────────────────────────────────────── */}
      <section className="mb-24">
        <div className="glass rounded-3xl p-10 md:p-14 text-center relative overflow-hidden glow-saffron">
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(255,107,53,0.08)] to-transparent pointer-events-none" />
          <div className="relative">
            <div className="flex justify-center mb-4">
              {[...Array(5)].map((_, i) => (
                <Star key={i} size={18} className="text-[var(--gold)] fill-[var(--gold)]" />
              ))}
            </div>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to scan your first dish?
            </h2>
            <p className="text-[var(--text-secondary)] mb-8 max-w-lg mx-auto">
              Create a profile to save your allergen preferences, or jump straight in and
              scan any Sri Lankan dish photo right now.
            </p>
            <Link
              href="/scan"
              className="btn-primary px-10 py-4 rounded-xl text-base font-semibold inline-flex items-center gap-2"
            >
              <ScanLine size={18} /> Scan a Dish Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
