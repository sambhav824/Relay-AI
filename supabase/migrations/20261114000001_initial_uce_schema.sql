-- Relay AI Universal Context Engine: initial cloud schema.
-- Every user-owned row carries user_id and is protected by RLS.
-- Composite foreign keys prevent cross-user parent/child relationships.

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  preferences jsonb not null default '{}'::jsonb,
  sync_enabled boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  unique (user_id, name)
);

create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid,
  type text not null default 'chat' check (type in ('chat','smart_routing','compare','team')),
  title text not null default 'New Conversation',
  model_id text,
  summary text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  foreign key (project_id, user_id) references public.projects(id, user_id) on delete cascade
);

create table public.messages (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  conversation_id uuid not null,
  role text not null check (role in ('user','assistant','system')),
  content text not null,
  token_count integer not null default 0 check (token_count >= 0),
  latency_ms double precision check (latency_ms is null or latency_ms >= 0),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  foreign key (conversation_id, user_id) references public.conversations(id, user_id) on delete cascade
);

create table public.memories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid,
  category text not null,
  content text not null,
  content_hash text,
  provenance text not null default 'USER_EXPLICIT',
  confidence real not null default 1.0 check (confidence between 0 and 1),
  is_approved boolean not null default true,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  foreign key (project_id, user_id) references public.projects(id, user_id) on delete cascade
);

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid,
  file_name text not null,
  storage_path text not null,
  file_hash text,
  mime_type text,
  file_size_bytes bigint check (file_size_bytes is null or file_size_bytes >= 0),
  status text not null default 'PENDING' check (status in ('PENDING','PROCESSING','INDEXED','FAILED','NEEDS_OCR')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  unique (user_id, storage_path),
  foreign key (project_id, user_id) references public.projects(id, user_id) on delete cascade
);

create table public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  document_id uuid not null,
  chunk_index integer not null check (chunk_index >= 0),
  content text not null,
  tsv_content tsvector generated always as (to_tsvector('english', coalesce(content,''))) stored,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  foreign key (document_id, user_id) references public.documents(id, user_id) on delete cascade,
  unique (document_id, chunk_index)
);

create table public.smart_routing_logs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  conversation_id uuid,
  message_id uuid,
  user_prompt text not null,
  classified_intent text,
  task_complexity text,
  routing_rationale text,
  selected_provider text,
  selected_model text,
  confidence_score real check (confidence_score is null or confidence_score between 0 and 1),
  created_at timestamptz not null default now(),
  foreign key (conversation_id, user_id) references public.conversations(id, user_id) on delete cascade,
  foreign key (message_id, user_id) references public.messages(id, user_id) on delete cascade
);

create table public.model_comparisons (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  conversation_id uuid,
  prompt text not null,
  created_at timestamptz not null default now(),
  foreign key (conversation_id, user_id) references public.conversations(id, user_id) on delete cascade
);

create table public.comparison_slots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  comparison_id uuid not null,
  slot_index integer not null check (slot_index >= 0),
  provider_id text not null,
  model_id text not null,
  response_text text,
  duration_ms double precision,
  token_count integer,
  foreign key (comparison_id, user_id) references public.model_comparisons(id, user_id) on delete cascade,
  unique (comparison_id, slot_index)
);

create table public.team_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  project_id uuid,
  team_name text not null,
  task_prompt text not null,
  status text not null default 'PENDING',
  final_deliverable text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (id, user_id),
  foreign key (project_id, user_id) references public.projects(id, user_id) on delete cascade
);

create table public.team_step_records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  team_run_id uuid not null,
  step_id text not null,
  agent_role text not null,
  output_text text,
  revision_index integer not null default 0,
  critic_feedback text,
  created_at timestamptz not null default now(),
  foreign key (team_run_id, user_id) references public.team_runs(id, user_id) on delete cascade
);

create index conversations_user_updated_idx on public.conversations(user_id, updated_at desc);
create index messages_conversation_created_idx on public.messages(conversation_id, created_at);
create index memories_user_project_idx on public.memories(user_id, project_id);
create index documents_user_project_idx on public.documents(user_id, project_id);
create index document_chunks_tsv_idx on public.document_chunks using gin(tsv_content);
create index routing_logs_user_created_idx on public.smart_routing_logs(user_id, created_at desc);
create index comparisons_user_created_idx on public.model_comparisons(user_id, created_at desc);
create index team_runs_user_created_idx on public.team_runs(user_id, created_at desc);

-- Keep updated_at consistent on mutable tables.
do $$
declare t text;
begin
  foreach t in array array['profiles','projects','conversations','memories','documents','team_runs'] loop
    execute format('create trigger %I before update on public.%I for each row execute function public.set_updated_at()', 'set_updated_at_' || t, t);
  end loop;
end $$;

-- Enable RLS on every user-owned table.
alter table public.profiles enable row level security;
alter table public.projects enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.memories enable row level security;
alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;
alter table public.smart_routing_logs enable row level security;
alter table public.model_comparisons enable row level security;
alter table public.comparison_slots enable row level security;
alter table public.team_runs enable row level security;
alter table public.team_step_records enable row level security;

-- Direct user_id policies; ownership is additionally enforced by composite FKs.
do $$
declare t text;
begin
  foreach t in array array['projects','conversations','messages','memories','documents','document_chunks','smart_routing_logs','model_comparisons','comparison_slots','team_runs','team_step_records'] loop
    execute format('create policy %I on public.%I for all to authenticated using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()))', t || '_owner_all', t);
  end loop;
end $$;

create policy profiles_owner_all on public.profiles
for all to authenticated
using (id = (select auth.uid()))
with check (id = (select auth.uid()));

-- Private document bucket. Object keys begin with the authenticated user's UUID.
insert into storage.buckets (id, name, public)
values ('relay-documents', 'relay-documents', false)
on conflict (id) do update set public = false;

create policy relay_documents_insert_own on storage.objects
for insert to authenticated
with check (
  bucket_id = 'relay-documents'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);
create policy relay_documents_select_own on storage.objects
for select to authenticated
using (
  bucket_id = 'relay-documents'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);
create policy relay_documents_delete_own on storage.objects
for delete to authenticated
using (
  bucket_id = 'relay-documents'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);

-- Account deletion is self-service only; cascades remove relational data.
create or replace function public.delete_user_account()
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare v_user_id uuid := auth.uid();
begin
  if v_user_id is null then
    raise exception 'Authentication required';
  end if;
  delete from auth.users where id = v_user_id;
end;
$$;
revoke all on function public.delete_user_account() from public, anon;
grant execute on function public.delete_user_account() to authenticated;
