import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://fxvpdyizlgjxnerjckjc.supabase.co";
const supabasePublishableKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY || "";

let client: SupabaseClient | null = null;

if (supabaseUrl && supabasePublishableKey) {
  try {
    client = createClient(supabaseUrl, supabasePublishableKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
      },
    });
  } catch (err) {
    console.error("Failed to initialize Supabase client:", err);
  }
}

export const supabase = client;
export const isSupabaseConfigured = Boolean(supabaseUrl && supabasePublishableKey);
