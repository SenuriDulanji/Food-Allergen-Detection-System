"use client";

import { useState, useEffect } from "react";
import {
  User, Save, Trash2, Plus, X, CheckCircle2, AlertTriangle,
  Loader2, Shield, Edit2, LogIn,
} from "lucide-react";
import { registerUser, getUser, updateUser, deleteUser, listAllergens } from "@/lib/api";
import type { UserResponse } from "@/lib/types";

const SRI_LANKAN_PROVINCES = [
  "Western", "Central", "Southern", "Northern", "Eastern",
  "North Western", "North Central", "Uva", "Sabaragamuwa",
];
const GENDERS = ["Male", "Female"];
const BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const DIETARY_PATTERNS = [
  "Omnivore (Eats everything)",
  "Vegetarian (Eats dairy/eggs, no meat)",
  "Vegan (No animal products)"
];
const WORK_ENVIRONMENTS = [
  "Dust or strong pollutants",
  "Extreme heat or direct sunlight",
  "Continuous air-conditioning",
  "Chemicals or strong fumes",
  "Agricultural or plant materials",
  "Standard home environment (Work from Home / Remote)",
  "None of the above"
];
const OUTSIDE_FOOD_OPTIONS = [
  { value: "4", label: "Daily" },
  { value: "3", label: "3–4 times a week" },
  { value: "2", label: "1–2 times a week" },
  { value: "1", label: "Rarely (a few times a month)" },
  { value: "0", label: "Never / I only eat home-cooked food" }
];
const MEDICAL_CONDITIONS = ["Asthma", "Eczema", "Food Allergy", "None of these", "Not sure"];
const FAMILY_HISTORY_OPTIONS = ["Beef", "Pork", "Sausage", "Red Meat", "Seafood", "Dairy", "Nuts", "Egg", "Tomato", "Pineapple", "Avocado", "Breadfruit", "Flour"];

const ALLERGEN_GROUPS: Record<string, string[]> = {
  "Common": ["gluten", "milk_dairy", "eggs", "soy", "peanuts", "tree_nuts", "coconut"],
  "Meats": ["beef", "pork", "lamb_mutton"],
  "Seafood": ["fish", "crustaceans", "molluscs", "shellfish", "prawns_shrimp", "crab", "cuttlefish_squid", "dry_fish_sprats", "tuna_balaya_kelawalla"],
  "Spices & Others": ["celery", "lupin", "mustard", "sesame", "sulphites", "spices", "spicy_oily", "fenugreek", "cumin"],
};

const ALLERGEN_LABELS: Record<string, string> = {
  "milk_dairy": "Cow's Milk / Dairy",
  "prawns_shrimp": "Prawns / Shrimp",
  "cuttlefish_squid": "Cuttlefish / Squid (Dallo)",
  "dry_fish_sprats": "Dry Fish / Sprats (Karawala)",
  "tuna_balaya_kelawalla": "Tuna / Balaya / Kelawalla",
  "lamb_mutton": "Lamb / Mutton",
  "coconut": "Coconut (Milk/Oil)",
  "spicy_oily": "Spicy / Oily Foods",
  "tree_nuts": "Tree Nuts (Cashew...)"
};

// ─── Toast notification ───────────────────────────────────────────────────────
function Toast({ msg, type }: { msg: string; type: "success" | "error" }) {
  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-2 px-5 py-3 rounded-xl shadow-2xl fade-in text-sm font-medium ${
      type === "success"
        ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-300"
        : "bg-red-500/20 border border-red-500/30 text-red-300"
    }`}>
      {type === "success" ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
      {msg}
    </div>
  );
}

// ─── Profile Display Card ─────────────────────────────────────────────────────
function ProfileCard({ user, onEdit, onDelete }: {
  user: UserResponse;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const statItems = [
    { label: "User ID", value: `#${user.id}` },
    { label: "Age", value: user.age ? `${user.age} yrs` : "—" },
    { label: "Gender", value: user.gender ?? "—" },
    { label: "Province", value: user.province ?? "—" },
    { label: "Blood Type", value: user.blood_type ?? "—" },
    { label: "Dietary Pattern", value: user.dietary_pattern ? user.dietary_pattern.split("(")[0].trim() : "—" },
  ];

  return (
    <div className="fade-in w-full space-y-5">
      {/* ── Header Card ── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="bg-gradient-to-r from-[rgba(255,107,53,0.12)] to-[rgba(245,158,11,0.06)] px-6 md:px-8 py-6 flex items-center justify-between">
          <div className="flex items-center gap-5">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#FF6B35] to-[#F59E0B] flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-orange-500/20">
              {(user.full_name ?? user.email)[0].toUpperCase()}
            </div>
            <div>
              <h2 className="font-bold text-xl md:text-2xl">{user.full_name ?? "—"}</h2>
              <p className="text-sm text-[var(--text-secondary)]">{user.email}</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={onEdit} className="btn-ghost px-4 py-2 rounded-xl flex items-center gap-2 text-sm font-medium">
              <Edit2 size={15} /> Edit
            </button>
            <button onClick={onDelete} className="btn-ghost px-4 py-2 rounded-xl flex items-center gap-2 text-sm font-medium hover:border-red-500/50 hover:text-red-400">
              <Trash2 size={15} /> Delete
            </button>
          </div>
        </div>
      </div>

      {/* ── Stats Grid ── */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {statItems.map(({ label, value }) => (
          <div key={label} className="glass rounded-2xl p-5">
            <p className="text-xs text-[var(--text-muted)] mb-1.5 uppercase tracking-wider font-medium">{label}</p>
            <p className="font-semibold text-base capitalize">{value}</p>
          </div>
        ))}
      </div>

      {/* ── Health & Lifestyle ── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="flex items-center gap-3 px-6 md:px-8 py-4 border-b border-white/5 bg-white/[0.02]">
          <Shield size={16} className="text-[var(--saffron)]" />
          <h3 className="text-sm font-bold uppercase tracking-wider">Health & Lifestyle</h3>
        </div>
        <div className="px-6 md:px-8 py-6 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
          {[
            { label: "Work Environment", value: user.work_env ?? "—" },
            { label: "Outside Food", value: user.outside_food_frequency !== null && user.outside_food_frequency !== undefined
              ? ["Never", "Rarely", "1–2x/week", "3–4x/week", "Daily"][user.outside_food_frequency] ?? "—"
              : "—" },
            { label: "Lactose Intolerance", value: user.lactose_intolerance ? "Yes" : user.lactose_intolerance === false ? "No" : "—" },
            { label: "Personal Allergy History", value: user.personal_allergy_history ? "Yes" : user.personal_allergy_history === false ? "No" : "—" },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
              <span className="text-sm text-[var(--text-secondary)]">{label}</span>
              <span className="text-sm font-medium capitalize">{value}</span>
            </div>
          ))}
          {user.medical_conditions && user.medical_conditions.length > 0 && (
            <div className="md:col-span-2 pt-2">
              <span className="text-sm text-[var(--text-secondary)] mr-3">Medical Conditions:</span>
              <span className="inline-flex flex-wrap gap-2 mt-1">
                {user.medical_conditions.map((c) => (
                  <span key={c} className="chip chip-warning capitalize">{c}</span>
                ))}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ── Allergen Profile ── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 md:px-8 py-4 border-b border-white/5 bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <Shield size={16} className="text-red-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Allergen Profile</h3>
          </div>
          <span className="text-xs font-semibold bg-red-500/10 text-red-400 px-3 py-1 rounded-lg">
            {user.allergens.length} allergen{user.allergens.length !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="px-6 md:px-8 py-6">
          <div className="flex flex-wrap gap-2.5">
            {user.allergens.length > 0
              ? user.allergens.map((a) => (
                  <span key={a} className="chip chip-danger capitalize text-sm px-3 py-1.5">{a.replace(/_/g, " ")}</span>
                ))
              : <span className="text-sm text-[var(--text-muted)]">No allergens saved. Edit your profile to add some.</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Profile Page ────────────────────────────────────────────────────────
export default function ProfilePage() {
  const [view, setView] = useState<"login" | "register" | "profile">("login");
  const [user, setUser] = useState<UserResponse | null>(null);
  const [allergenOptions, setAllergenOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; type: "success" | "error" } | null>(null);

  // Form state
  const [lookupId, setLookupId] = useState("");
  const [customAllergen, setCustomAllergen] = useState("");
  const [form, setForm] = useState({
    email: "", password: "", full_name: "", age: "", gender: "", province: "",
    blood_type: "", dietary_pattern: "", outside_food_frequency: "", work_env: "",
  });
  const [selectedAllergens, setSelectedAllergens] = useState<string[]>([]);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [lactoseIntolerance, setLactoseIntolerance] = useState(false);
  const [personalAllergyHistory, setPersonalAllergyHistory] = useState(false);
  const [familyHasHistory, setFamilyHasHistory] = useState(false);
  const [familyAsthma, setFamilyAsthma] = useState(false);
  const [familyEczema, setFamilyEczema] = useState(false);
  const [selectedFamilyAllergies, setSelectedFamilyAllergies] = useState<string[]>([]);


  useEffect(() => {
    listAllergens().then(setAllergenOptions).catch(() => {});
    const savedId = localStorage.getItem("allerscan_user_id");
    if (savedId) {
      setLookupId(savedId);
    }
  }, []);

  const showToast = (msg: string, type: "success" | "error") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const handleLoad = async () => {
    const id = parseInt(lookupId, 10);
    if (!id) return;
    setLoading(true);
    try {
      const u = await getUser(id);
      setUser(u);
      setView("profile");
      localStorage.setItem("allerscan_user_id", String(id));
      showToast("Profile loaded!", "success");
    } catch {
      showToast("User not found. Check your ID.", "error");
      localStorage.removeItem("allerscan_user_id");
      setLookupId("");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    setLoading(true);
    try {
      const u = await registerUser({
        email: form.email,
        password: form.password,
        full_name: form.full_name || undefined,
        age: form.age ? parseInt(form.age, 10) : undefined,
        gender: form.gender || undefined,
        province: form.province || undefined,
        blood_type: form.blood_type || undefined,
        dietary_pattern: form.dietary_pattern || undefined,
        medical_conditions: selectedConditions,
        lactose_intolerance: lactoseIntolerance,
        outside_food_frequency: form.outside_food_frequency ? parseInt(form.outside_food_frequency, 10) : undefined,
        personal_allergy_history: personalAllergyHistory,
        work_env: form.work_env || undefined,
        family_has_history: familyHasHistory,
        family_asthma: familyAsthma,
        family_eczema: familyEczema,
        family_allergies: selectedFamilyAllergies,
        allergens: selectedAllergens,
      });
      setUser(u);
      setView("profile");
      localStorage.setItem("allerscan_user_id", String(u.id));
      showToast(`Welcome, ${u.full_name ?? u.email}!`, "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Registration failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async () => {
    if (!user) return;
    setLoading(true);
    try {
      const updated = await updateUser(user.id, {
        email: form.email || undefined,
        full_name: form.full_name || undefined,
        age: form.age ? parseInt(form.age, 10) : undefined,
        gender: form.gender || undefined,
        province: form.province || undefined,
        blood_type: form.blood_type || undefined,
        dietary_pattern: form.dietary_pattern || undefined,
        medical_conditions: selectedConditions,
        lactose_intolerance: lactoseIntolerance,
        outside_food_frequency: form.outside_food_frequency ? parseInt(form.outside_food_frequency, 10) : undefined,
        personal_allergy_history: personalAllergyHistory,
        work_env: form.work_env || undefined,
        family_has_history: familyHasHistory,
        family_asthma: familyAsthma,
        family_eczema: familyEczema,
        family_allergies: selectedFamilyAllergies,
        allergens: selectedAllergens,
      });
      setUser(updated);
      setView("profile");
      showToast("Profile updated!", "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Update failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!user || !confirm("Delete your account permanently?")) return;
    setLoading(true);
    try {
      await deleteUser(user.id);
      setUser(null);
      setView("register");
      localStorage.removeItem("allerscan_user_id");
      showToast("Account deleted.", "success");
    } catch {
      showToast("Delete failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const startEdit = () => {
    if (!user) return;
    setForm({
      email: user.email,
      password: "",
      full_name: user.full_name ?? "",
      age: user.age ? String(user.age) : "",
      gender: user.gender ?? "",
      province: user.province ?? "",
      blood_type: user.blood_type ?? "",
      dietary_pattern: user.dietary_pattern ?? "",
      work_env: user.work_env ?? "",
      outside_food_frequency: user.outside_food_frequency !== null && user.outside_food_frequency !== undefined ? String(user.outside_food_frequency) : "",
    });
    setSelectedAllergens([...user.allergens]);
    setSelectedConditions(user.medical_conditions ? [...user.medical_conditions] : []);
    setLactoseIntolerance(user.lactose_intolerance ?? false);
    setPersonalAllergyHistory(user.personal_allergy_history ?? false);
    setFamilyHasHistory(user.family_has_history ?? false);
    setFamilyAsthma(user.family_asthma ?? false);
    setFamilyEczema(user.family_eczema ?? false);
    setSelectedFamilyAllergies(user.family_allergies ? [...user.family_allergies] : []);
    setView("register"); // reuse the form
  };

  const toggleAllergen = (a: string) =>
    setSelectedAllergens((prev) => prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]);

  const toggleCondition = (c: string) =>
    setSelectedConditions((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]);

  const toggleFamilyAllergy = (a: string) =>
    setSelectedFamilyAllergies((prev) => prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]);

  const inputClass = "input-field w-full px-4 py-2.5 rounded-xl text-sm";
  const isEditing = view === "register" && !!user;

  return (
    <div className="min-h-screen flex items-center justify-center py-12">
      <div className="max-w-3xl w-full mx-auto px-6 flex flex-col gap-8">
      {toast && <Toast msg={toast.msg} type={toast.type} />}

      <div className="text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
          <span className="gradient-text">My</span> Profile
        </h1>
        <p className="text-lg text-[var(--text-secondary)]">
          Save your allergen profile to personalise every scan.
        </p>
      </div>

      {/* Tab switcher */}
      {!user && (
        <div className="flex gap-2 glass rounded-2xl p-1.5 w-full">
          {(["login", "register"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setView(tab)}
              className={`flex-1 py-3 rounded-xl text-base font-semibold transition-all capitalize ${
                view === tab
                  ? "bg-gradient-to-r from-[#FF6B35] to-[#E5531A] text-white shadow-md"
                  : "text-[var(--text-secondary)] hover:text-white"
              }`}
            >
              {tab === "login" ? "Load Profile" : "Create Account"}
            </button>
          ))}
        </div>
      )}

      {/* ── Profile View ── */}
      {view === "profile" && user && (
        <ProfileCard
          user={user}
          onEdit={startEdit}
          onDelete={handleDelete}
        />
      )}

      {/* ── Load Profile ── */}
      {view === "login" && (
        <div className="glass rounded-2xl p-8 md:p-10 space-y-6 fade-in w-full">
          <div className="flex items-center justify-center gap-3 mb-4">
            <LogIn size={24} className="text-[var(--saffron)]" />
            <h2 className="text-xl font-bold">Load Existing Profile</h2>
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
              Your User ID
            </label>
            <input
              type="number"
              placeholder="e.g. 1"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
              className={`${inputClass} w-full px-5 py-4 text-lg`}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            />
            <p className="text-sm text-[var(--text-muted)] mt-3 text-center">
              Your ID was shown when you registered.
            </p>
          </div>
          <button
            onClick={handleLoad}
            disabled={!lookupId || loading}
            className="btn-primary w-full py-4 rounded-2xl text-lg font-semibold flex items-center justify-center gap-2 transition-transform hover:scale-[1.02]"
          >
            {loading ? <Loader2 size={20} className="spin" /> : <LogIn size={20} />}
            Load Profile
          </button>
          <p className="text-center text-sm text-[var(--text-muted)] pt-2">
            Don&apos;t have an account?{" "}
            <button className="text-[var(--saffron)] font-medium hover:underline ml-1" onClick={() => setView("register")}>
              Create one free
            </button>
          </p>
        </div>
      )}

      {/* ── Register / Edit Form ── */}
      {(view === "register") && (
        <div className="fade-in w-full space-y-6">

          {/* ── Step 1: Account Info ── */}
          <div className="glass rounded-2xl overflow-hidden">
            <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5 bg-white/[0.02]">
              <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--saffron)]/15 text-[var(--saffron)] text-xs font-bold">1</span>
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">Account Credentials</h3>
            </div>
            <div className="p-6 md:p-8 space-y-5">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Email *</label>
                <input type="email" placeholder="you@example.com" value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })} className={`${inputClass} w-full`} />
              </div>
              {!isEditing && (
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Password * <span className="text-[var(--text-muted)] font-normal">(min 6 chars)</span></label>
                  <input type="password" placeholder="••••••" value={form.password}
                    onChange={(e) => setForm({ ...form, password: e.target.value })} className={`${inputClass} w-full`} />
                </div>
              )}
            </div>
          </div>

          {/* ── Step 2: Personal Details ── */}
          <div className="glass rounded-2xl overflow-hidden">
            <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5 bg-white/[0.02]">
              <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--saffron)]/15 text-[var(--saffron)] text-xs font-bold">2</span>
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">Personal Details</h3>
            </div>
            <div className="p-6 md:p-8 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Full Name</label>
                  <input type="text" placeholder="Kamal Perera" value={form.full_name}
                    onChange={(e) => setForm({ ...form, full_name: e.target.value })} className={`${inputClass} w-full`} />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Age (in years)</label>
                  <input type="number" placeholder="25" value={form.age}
                    onChange={(e) => setForm({ ...form, age: e.target.value })} className={`${inputClass} w-full`} />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Gender</label>
                  <select value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}
                    className={`${inputClass} w-full appearance-none`}>
                    <option value="">Select…</option>
                    {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Blood Type</label>
                  <select value={form.blood_type} onChange={(e) => setForm({ ...form, blood_type: e.target.value })}
                    className={`${inputClass} w-full appearance-none`}>
                    <option value="">Select…</option>
                    {BLOOD_TYPES.map((b) => <option key={b} value={b}>{b}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Province</label>
                <select value={form.province} onChange={(e) => setForm({ ...form, province: e.target.value })}
                  className={`${inputClass} w-full appearance-none md:w-1/2`}>
                  <option value="">Select…</option>
                  {SRI_LANKAN_PROVINCES.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* ── Step 3: Diet & Lifestyle ── */}
          <div className="glass rounded-2xl overflow-hidden">
            <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5 bg-white/[0.02]">
              <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--saffron)]/15 text-[var(--saffron)] text-xs font-bold">3</span>
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">Diet & Lifestyle</h3>
            </div>
            <div className="p-6 md:p-8 space-y-6">
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Dietary Pattern</label>
                <select value={form.dietary_pattern} onChange={(e) => setForm({ ...form, dietary_pattern: e.target.value })}
                  className={`${inputClass} w-full appearance-none`}>
                  <option value="">Select…</option>
                  {DIETARY_PATTERNS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Outside Food Consumption</label>
                <select value={form.outside_food_frequency} onChange={(e) => setForm({ ...form, outside_food_frequency: e.target.value })}
                  className={`${inputClass} w-full appearance-none`}>
                  <option value="">Select frequency…</option>
                  {OUTSIDE_FOOD_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Work Environment</label>
                <select value={form.work_env} onChange={(e) => setForm({ ...form, work_env: e.target.value })}
                  className={`${inputClass} w-full appearance-none`}>
                  <option value="">Select…</option>
                  {WORK_ENVIRONMENTS.map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* ── Step 4: Health Profile ── */}
          <div className="glass rounded-2xl overflow-hidden">
            <div className="flex items-center gap-3 px-6 py-4 border-b border-white/5 bg-white/[0.02]">
              <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--saffron)]/15 text-[var(--saffron)] text-xs font-bold">4</span>
              <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">Health Profile</h3>
            </div>
            <div className="p-6 md:p-8 space-y-8">
              {/* Medical Conditions */}
              <div>
                <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">Medical Conditions</label>
               <div className="flex flex-wrap gap-3">
                  {MEDICAL_CONDITIONS.map((c) => (
                    <button key={c} onClick={() => toggleCondition(c)}
                      className={`allergen-toggle px-4 py-2.5 rounded-xl text-sm font-medium capitalize transition-colors ${selectedConditions.includes(c) ? "selected" : ""}`}>
                      {c}
                    </button>
                  ))}
                </div>
              </div>

              {/* Checkboxes */}
              <div className="space-y-4 bg-white/[0.02] rounded-xl p-5 border border-white/5">
                <label className="flex items-center gap-4 cursor-pointer text-sm text-[var(--text-secondary)] hover:text-white transition-colors py-1">
                  <input type="checkbox" checked={lactoseIntolerance} onChange={(e) => setLactoseIntolerance(e.target.checked)} className="rounded border-white/20 bg-white/5 w-5 h-5 text-[var(--saffron)] focus:ring-[var(--saffron)] focus:ring-offset-0 transition-colors shrink-0" />
                  Have you been diagnosed with lactose intolerance?
                </label>
                <div className="border-t border-white/5" />
                <label className="flex items-center gap-4 cursor-pointer text-sm text-[var(--text-secondary)] hover:text-white transition-colors py-1">
                  <input type="checkbox" checked={personalAllergyHistory} onChange={(e) => setPersonalAllergyHistory(e.target.checked)} className="rounded border-white/20 bg-white/5 w-5 h-5 text-[var(--saffron)] focus:ring-[var(--saffron)] focus:ring-offset-0 transition-colors shrink-0" />
                  Have you personally experienced an allergic reaction to food?
                </label>
              </div>

              {/* Family Medical History */}
              <div className="bg-white/[0.02] rounded-xl p-5 border border-white/5">
                <label className="flex items-center gap-4 cursor-pointer text-sm font-medium text-white mb-1 py-1">
                  <input type="checkbox" checked={familyHasHistory} onChange={(e) => setFamilyHasHistory(e.target.checked)} className="rounded border-white/20 bg-white/5 w-5 h-5 text-[var(--saffron)] focus:ring-[var(--saffron)] focus:ring-offset-0 transition-colors shrink-0" />
                  Family History of Allergies
                </label>
               
                {familyHasHistory && (
                  <div className="ml-9 mt-4 space-y-5 fade-in border-t border-white/5 pt-5">
                    <div className="flex flex-wrap gap-5">
                      <label className="flex items-center gap-3 cursor-pointer text-sm text-[var(--text-secondary)] hover:text-white transition-colors">
                        <input type="checkbox" checked={familyAsthma} onChange={(e) => setFamilyAsthma(e.target.checked)} className="rounded border-white/20 bg-white/5 w-4 h-4 text-[var(--saffron)]" />
                        Family Asthma
                      </label>
                      <label className="flex items-center gap-3 cursor-pointer text-sm text-[var(--text-secondary)] hover:text-white transition-colors">
                        <input type="checkbox" checked={familyEczema} onChange={(e) => setFamilyEczema(e.target.checked)} className="rounded border-white/20 bg-white/5 w-4 h-4 text-[var(--saffron)]" />
                        Family Eczema
                      </label>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-[var(--text-secondary)] mb-3">Family Food Allergies</label>
                      <div className="flex flex-wrap gap-2.5">
                        {FAMILY_HISTORY_OPTIONS.map((a) => (
                          <button key={a} onClick={() => toggleFamilyAllergy(a)}
                            className={`allergen-toggle px-3 py-1.5 rounded-xl text-xs font-medium capitalize transition-colors ${selectedFamilyAllergies.includes(a) ? "selected" : ""}`}>
                            {a}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Step 5: Allergens ── */}
          <div className="glass rounded-2xl overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--saffron)]/15 text-[var(--saffron)] text-xs font-bold">5</span>
                <h3 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">Your Allergens</h3>
              </div>
              {selectedAllergens.length > 0 && (
                <button className="text-xs font-semibold bg-red-500/10 text-red-400 hover:bg-red-500/20 px-3 py-1.5 rounded-lg transition-colors"
                  onClick={() => setSelectedAllergens([])}>Clear all ({selectedAllergens.length})</button>
              )}
            </div>
            <div className="p-6 md:p-8">
            
              <div className="space-y-6">
                {Object.entries(ALLERGEN_GROUPS).map(([groupName, allergensInGroup]) => {
                  const available = allergensInGroup.filter(a => allergenOptions.includes(a));
                  if (available.length === 0) return null;
                  return (
                    <div key={groupName} className="bg-white/[0.02] rounded-xl p-4 border border-white/5">
                      <p className="text-xs text-[var(--saffron)] mb-3 uppercase tracking-wider font-bold">{groupName}</p>
                      <div className="flex flex-wrap gap-2.5">
                        {available.map((a) => (
                          <button key={a} onClick={() => toggleAllergen(a)}
                            className={`allergen-toggle px-4 py-2 rounded-xl text-sm font-medium capitalize transition-colors ${selectedAllergens.includes(a) ? "selected" : ""}`}>
                            {ALLERGEN_LABELS[a] || a.replace(/_/g, " ")}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
                
                {/* Fallback for allergens not in groups */}
                {allergenOptions.filter(a => !Object.values(ALLERGEN_GROUPS).flat().includes(a)).length > 0 && (
                  <div className="bg-white/[0.02] rounded-xl p-4 border border-white/5">
                    <p className="text-xs text-[var(--saffron)] mb-3 uppercase tracking-wider font-bold">Additional</p>
                    <div className="flex flex-wrap gap-2.5">
                      {allergenOptions.filter(a => !Object.values(ALLERGEN_GROUPS).flat().includes(a)).map((a) => (
                        <button key={a} onClick={() => toggleAllergen(a)}
                          className={`allergen-toggle px-4 py-2 rounded-xl text-sm font-medium capitalize transition-colors ${selectedAllergens.includes(a) ? "selected" : ""}`}>
                          {ALLERGEN_LABELS[a] || a.replace(/_/g, " ")}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Custom Allergens Display */}
                {selectedAllergens.filter(a => !allergenOptions.includes(a)).length > 0 && (
                  <div className="bg-white/[0.02] rounded-xl p-4 border border-white/5">
                    <p className="text-xs text-[var(--saffron)] mb-3 uppercase tracking-wider font-bold">Custom Additions</p>
                    <div className="flex flex-wrap gap-2.5">
                      {selectedAllergens.filter(a => !allergenOptions.includes(a)).map((a) => (
                        <button key={a} onClick={(e) => { e.preventDefault(); toggleAllergen(a); }}
                          className="allergen-toggle selected px-4 py-2 rounded-xl text-sm font-medium capitalize transition-colors flex items-center gap-1.5">
                          {a.replace(/_/g, " ")} <X size={14} className="opacity-70" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Add Custom Allergen Input */}
                <div className="pt-4 border-t border-white/5 mt-2">
                  <p className="text-sm font-medium text-[var(--text-secondary)] mb-3">Don&apos;t see your allergy? Add it manually:</p>
                  <div className="flex gap-3">
                    <input
                      type="text"
                      placeholder="e.g. kiwi, avocado..."
                      value={customAllergen}
                      onChange={(e) => setCustomAllergen(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          if (customAllergen.trim() && !selectedAllergens.includes(customAllergen.trim().toLowerCase())) {
                            toggleAllergen(customAllergen.trim().toLowerCase());
                            setCustomAllergen("");
                          }
                        }
                      }}
                      className={`${inputClass} flex-1`}
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        if (customAllergen.trim() && !selectedAllergens.includes(customAllergen.trim().toLowerCase())) {
                          toggleAllergen(customAllergen.trim().toLowerCase());
                          setCustomAllergen("");
                        }
                      }}
                      className="btn-ghost bg-white/5 hover:bg-white/10 px-6 py-2 rounded-xl font-medium transition-colors"
                    >
                      Add
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── Actions ── */}
          <div className="glass rounded-2xl p-6 flex flex-col sm:flex-row gap-4">
            <button
              onClick={isEditing ? handleUpdate : handleRegister}
              disabled={!form.email || (!isEditing && !form.password) || loading}
              className="btn-primary flex-1 py-4 rounded-2xl text-lg font-semibold flex items-center justify-center gap-2 transition-transform hover:scale-[1.02]"
            >
              {loading ? <Loader2 size={20} className="spin" /> : isEditing ? <Save size={20} /> : <Plus size={20} />}
              {isEditing ? "Save Changes" : "Create Account"}
            </button>
            
            {isEditing && (
              <button onClick={() => setView("profile")} className="btn-ghost px-8 py-4 rounded-2xl text-lg font-medium flex items-center justify-center gap-2 transition-colors">
                <X size={20} /> Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
</div>
  );
}
