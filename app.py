import os
import sys
from PySide6.QtWidgets import QApplication
from src.context.auth.manager import AuthManager
from src.context.db.connection import initialize_database
from src.context.repository import ContextRepository
from src.context.sync.queue import SyncQueueManager
from src.context.sync.worker import SyncWorker
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Relay AI")
    app.setOrganizationName("Relay AI")

    # Initialize local context DB and sync outbox
    db = initialize_database()
    conn = db.get_connection()
    sync_queue = SyncQueueManager(conn)
    auth_manager = AuthManager()
    try:
        auth_manager.restore_session()
    except Exception:
        pass

    current_user_id = auth_manager.user.id if auth_manager.user else None
    context_repo = ContextRepository(conn, sync_queue=sync_queue, default_user_id=current_user_id)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY")

    sync_worker = None
    if supabase_url and supabase_anon_key:
        sync_worker = SyncWorker(
            queue=sync_queue,
            auth_manager=auth_manager,
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
        )
        sync_worker.start()

    window = MainWindow(
        sync_worker=sync_worker,
        auth_manager=auth_manager,
        sync_queue=sync_queue,
        context_repo=context_repo,
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
