-- Rollback for 20261114000001_initial_uce_schema.sql.
-- Storage bucket is intentionally retained (private) so rollback cannot orphan/delete user files.
drop policy if exists relay_documents_insert_own on storage.objects;
drop policy if exists relay_documents_select_own on storage.objects;
drop policy if exists relay_documents_delete_own on storage.objects;

drop function if exists public.delete_user_account();
drop function if exists public.set_updated_at();

drop table if exists public.team_step_records cascade;
drop table if exists public.team_runs cascade;
drop table if exists public.comparison_slots cascade;
drop table if exists public.model_comparisons cascade;
drop table if exists public.smart_routing_logs cascade;
drop table if exists public.document_chunks cascade;
drop table if exists public.documents cascade;
drop table if exists public.memories cascade;
drop table if exists public.messages cascade;
drop table if exists public.conversations cascade;
drop table if exists public.projects cascade;
drop table if exists public.profiles cascade;
