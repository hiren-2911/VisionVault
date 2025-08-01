sudo unlink /tmp/supervisor.sock
sudo rm /tmp/supervisor.sock

sudo systemctl restart supervisor
sudo service supervisor restart

sudo supervisorctl  shutdown
sudo supervisorctl  stop all

