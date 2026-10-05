import type { ScanResponse, UserCreate, UserResponse, UserUpdate } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

// ─── Scan ─────────────────────────────────────────────────────────────────────

export async function scanDish(
  imageFile: File,
  userAllergens: string[],
  userId?: number
): Promise<ScanResponse> {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("user_allergens", JSON.stringify(userAllergens));
  if (userId !== undefined) form.append("user_id", String(userId));

  const res = await fetch(`${BASE_URL}/scan/`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `Scan failed (${res.status})`);
  }
  return res.json();
}

export async function listAllergens(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/scan/allergens`);
  const data = await res.json();
  return data.allergen_categories ?? [];
}

export async function listDishes(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/scan/dishes`);
  const data = await res.json();
  return data.dishes ?? [];
}

// ─── Users ────────────────────────────────────────────────────────────────────

export async function registerUser(payload: UserCreate): Promise<UserResponse> {
  const res = await fetch(`${BASE_URL}/users/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `Registration failed (${res.status})`);
  }
  return res.json();
}

export async function getUser(userId: number): Promise<UserResponse> {
  const res = await fetch(`${BASE_URL}/users/${userId}`);
  if (!res.ok) throw new Error(`User not found (${res.status})`);
  return res.json();
}

export async function updateUser(userId: number, payload: UserUpdate): Promise<UserResponse> {
  const res = await fetch(`${BASE_URL}/users/${userId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `Update failed (${res.status})`);
  }
  return res.json();
}

export async function deleteUser(userId: number): Promise<void> {
  const res = await fetch(`${BASE_URL}/users/${userId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}
